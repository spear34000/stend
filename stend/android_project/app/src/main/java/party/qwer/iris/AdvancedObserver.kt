package party.qwer.iris

import android.database.sqlite.SQLiteDatabase
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.runBlocking
import org.json.JSONObject

class AdvancedObserver(
    private val kakaoDb: KakaoDB,
    private val wsFlow: MutableSharedFlow<String>
) {
    fun installTriggers() {
        try {
            val db = kakaoDb.connection
            
            // 1. 통합 이벤트 테이블 생성 (enc 컬럼 추가)
            db.execSQL("""
                CREATE TABLE IF NOT EXISTS stend_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    target_id INTEGER,
                    data1 TEXT,
                    data2 TEXT,
                    enc INTEGER,
                    timestamp INTEGER
                )
            """)

            // 2. 닉네임 변경 트리거
            db.execSQL("DROP TRIGGER IF EXISTS trg_stend_nick_change")
            db.execSQL("""
                CREATE TRIGGER trg_stend_nick_change
                AFTER UPDATE ON db2.friends
                WHEN OLD.name != NEW.name
                BEGIN
                    INSERT INTO stend_events (event_type, target_id, data1, data2, enc, timestamp)
                    VALUES ('NICKNAME_CHANGE', NEW.id, OLD.name, NEW.name, NEW.enc, strftime('%s','now'));
                END
            """)

            // 3. 프로필 이미지 변경 트리거
            db.execSQL("DROP TRIGGER IF EXISTS trg_stend_profile_change")
            db.execSQL("""
                CREATE TRIGGER trg_stend_profile_change
                AFTER UPDATE ON db2.friends
                WHEN OLD.profile_image_url != NEW.profile_image_url
                BEGIN
                    INSERT INTO stend_events (event_type, target_id, data1, data2, enc, timestamp)
                    VALUES ('PROFILE_CHANGE', NEW.id, OLD.profile_image_url, NEW.profile_image_url, NEW.enc, strftime('%s','now'));
                END
            """)

            // 4. 상태 메시지 변경 트리거
            db.execSQL("DROP TRIGGER IF EXISTS trg_stend_status_change")
            db.execSQL("""
                CREATE TRIGGER trg_stend_status_change
                AFTER UPDATE ON db2.friends
                WHEN OLD.status_message != NEW.status_message
                BEGIN
                    INSERT INTO stend_events (event_type, target_id, data1, data2, enc, timestamp)
                    VALUES ('STATUS_CHANGE', NEW.id, OLD.status_message, NEW.status_message, NEW.enc, strftime('%s','now'));
                END
            """)

            // 5. 메시지 삭제 감지 트리거
            db.execSQL("DROP TRIGGER IF EXISTS trg_stend_msg_delete")
            db.execSQL("""
                CREATE TRIGGER trg_stend_msg_delete
                AFTER UPDATE ON chat_logs
                WHEN OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL
                BEGIN
                    INSERT INTO stend_events (event_type, target_id, data1, data2, enc, timestamp)
                    VALUES ('MESSAGE_DELETE', NEW._id, NEW.message, NEW.chat_id, NEW.type, strftime('%s','now'));
                END
            """)

            // 4. 메시지 가리기 감지
            db.execSQL("DROP TRIGGER IF EXISTS trg_stend_msg_hide")
            db.execSQL("""
                CREATE TRIGGER trg_stend_msg_hide
                AFTER UPDATE ON chat_logs
                WHEN (OLD.flags & 8 = 0) AND (NEW.flags & 8 != 0)
                BEGIN
                    INSERT INTO stend_events (event_type, target_id, data1, data2, enc, timestamp)
                    VALUES ('MESSAGE_HIDE', NEW._id, NEW.chat_id, NEW.user_id, 0, strftime('%s','now'));
                END
            """)

            // 5. 피드 메시지 감지
            db.execSQL("DROP TRIGGER IF EXISTS trg_stend_feed_event")
            db.execSQL("""
                CREATE TRIGGER trg_stend_feed_event
                AFTER INSERT ON chat_logs
                WHEN NEW.type = 26
                BEGIN
                    INSERT INTO stend_events (event_type, target_id, data1, data2, enc, timestamp)
                    VALUES ('FEED_EVENT', NEW._id, NEW.message, NEW.chat_id, 0, strftime('%s','now'));
                END
            """)
            
            println("[Stend] Advanced Triggers installed successfully.")
        } catch (e: Exception) {
            println("[Stend] Failed to install advanced triggers: ${e.message}")
        }
    }

    fun pollEvents() {
        try {
            val db = kakaoDb.connection
            db.rawQuery("SELECT * FROM stend_events ORDER BY id ASC", null).use { cursor ->
                while (cursor.moveToNext()) {
                    val type = cursor.getString(cursor.getColumnIndexOrThrow("event_type"))
                    val targetId = cursor.getLong(cursor.getColumnIndexOrThrow("target_id"))
                    val d1 = cursor.getString(cursor.getColumnIndexOrThrow("data1"))
                    val d2 = cursor.getString(cursor.getColumnIndexOrThrow("data2"))
                    val enc = cursor.getInt(cursor.getColumnIndexOrThrow("enc"))
                    val ts = cursor.getLong(cursor.getColumnIndexOrThrow("timestamp"))

                    val evt = JSONObject()
                    evt.put("type", "stend_event")
                    evt.put("event", type)
                    evt.put("target_id", targetId)
                    evt.put("timestamp", ts)

                    when(type) {
                        "NICKNAME_CHANGE" -> {
                            try {
                                val oldDec = KakaoDecrypt.decrypt(enc, d1, Configurable.botId)
                                val newDec = KakaoDecrypt.decrypt(enc, d2, Configurable.botId)
                                evt.put("from", oldDec)
                                evt.put("to", newDec)
                            } catch(e: Exception) {
                                evt.put("from", d1); evt.put("to", d2)
                            }
                        }
                        "MESSAGE_DELETE" -> {
                            try {
                                // For messages, we need the 'enc' from the 'v' column, but triggers can't easily parse JSON.
                                // We'll try common enc values or store 'v' in data1 instead of just message.
                                // For now, attempt decryption with default enc=2 (fallback to raw)
                                val msgDec = KakaoDecrypt.decrypt(2, d1, Configurable.botId)
                                evt.put("deleted_content", msgDec)
                            } catch(e: Exception) {
                                evt.put("deleted_content", d1)
                            }
                            evt.put("chat_id", d2)
                        }
                        "MESSAGE_HIDE" -> {
                             evt.put("chat_id", d1)
                             evt.put("user_id", d2)
                        }
                        "FEED_EVENT" -> {
                             try {
                                 val feedJson = JSONObject(d1)
                                 val feedType = feedJson.optInt("feedType")
                                 evt.put("feed_type", feedType)
                                 evt.put("chat_id", d2)
                                 
                                 val subEvent = when(feedType) {
                                     1 -> "JOIN"
                                     2 -> "LEAVE"
                                     6 -> "KICK"
                                     12 -> "DEMOTE"
                                     11 -> "PROMOTE"
                                     14 -> "HANDOVER"
                                     else -> "UNKNOWN_FEED"
                                 }
                                 evt.put("sub_event", subEvent)
                             } catch(e: Exception) {
                                 evt.put("sub_event", "PARSE_ERROR")
                             }
                        }
                        "PROFILE_CHANGE" -> {
                            evt.put("from", d1); evt.put("to", d2)
                        }
                        "STATUS_CHANGE" -> {
                             try {
                                 val oldDec = KakaoDecrypt.decrypt(enc, d1, Configurable.botId)
                                 val newDec = KakaoDecrypt.decrypt(enc, d2, Configurable.botId)
                                 evt.put("from", oldDec)
                                 evt.put("to", newDec)
                             } catch(e: Exception) {
                                 evt.put("from", d1); evt.put("to", d2)
                             }
                        }
                    }
                    
                    runBlocking {
                         wsFlow.emit(evt.toString())
                    }
                }
            }
            db.execSQL("DELETE FROM stend_events")
        } catch (e: Exception) { }
    }
}
