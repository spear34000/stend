package party.qwer.iris

import android.app.RemoteInput
import android.content.ComponentName
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.charset.StandardCharsets
import java.net.URLDecoder
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec

class StendGrandService(private val kakaoDB: KakaoDB) {

    /**
     * Get list of all chat rooms
     */
    fun getRoomList(): JSONArray {
        val sql = "SELECT id, type, active_member_count, last_log_id, private_meta FROM chat_rooms ORDER BY last_log_id DESC"
        val rows = kakaoDB.executeQuery(sql, null)
        val result = JSONArray()
        for (row in rows) {
            val dec = KakaoDB.decryptRow(row)
            val obj = JSONObject(dec)
            try {
                val meta = dec["private_meta"]
                if (meta != null && meta.startsWith("{")) {
                    val metaJson = JSONObject(meta)
                    obj.put("name", metaJson.optString("name", "Unknown Room"))
                }
            } catch (e: Exception) {}
            result.put(obj)
        }
        return result
    }

    /**
     * Get member list of a room
     */
    fun getMembers(chatId: Long): JSONArray {
        val isNew = kakaoDB.checkNewDb()
        val sql = if (isNew) {
            """
            SELECT user_id, nickname, profile_image_url, enc, member_type 
            FROM db2.open_chat_member 
            WHERE link_id = (SELECT link_id FROM chat_rooms WHERE id = ?)
            UNION
            SELECT id as user_id, name as nickname, profile_image_url, enc, 0 as member_type
            FROM db2.friends
            WHERE id IN (SELECT user_id FROM chat_logs WHERE chat_id = ? GROUP BY user_id)
            """.trimIndent()
        } else {
            "SELECT id as user_id, name as nickname, profile_image_url, enc, 0 as member_type FROM db2.friends"
        }
        
        val rows = kakaoDB.executeQuery(sql, arrayOf(chatId.toString(), chatId.toString()))
        val result = JSONArray()
        for (row in rows) {
            val dec = KakaoDB.decryptRow(row)
            result.put(JSONObject(dec))
        }
        return result
    }

    /**
     * Get message history
     */
    fun getHistory(chatId: Long, limit: Int = 100): JSONArray {
        val sql = "SELECT * FROM chat_logs WHERE chat_id = ? ORDER BY _id DESC LIMIT ?"
        val rows = kakaoDB.executeQuery(sql, arrayOf(chatId.toString(), limit.toString()))
        val result = JSONArray()
        for (row in rows) {
            val dec = KakaoDB.decryptRow(row)
            result.put(JSONObject(dec))
        }
        return result
    }

    /**
     * Search for messages in a room
     */
    fun search(chatId: Long, query: String, limit: Int = 100): JSONArray {
        val sql = "SELECT * FROM chat_logs WHERE chat_id = ? AND message LIKE ? ORDER BY _id DESC LIMIT ?"
        val rows = kakaoDB.executeQuery(sql, arrayOf(chatId.toString(), "%$query%", limit.toString()))
        val result = JSONArray()
        for (row in rows) {
            val dec = KakaoDB.decryptRow(row)
            result.put(JSONObject(dec))
        }
        return result
    }

    /**
     * Get detailed media/attachment info
     */
    fun getMediaInfo(logId: Long): JSONObject {
        val sql = "SELECT attachment, type FROM chat_logs WHERE _id = ?"
        val rows = kakaoDB.executeQuery(sql, arrayOf(logId.toString()))
        if (rows.isEmpty()) return JSONObject()
        val row = rows[0]
        val attachment = row["attachment"] ?: return JSONObject()
        val type = row["type"]?.toInt() ?: 0
        val res = JSONObject()
        res.put("log_id", logId)
        res.put("type", type)
        try {
            if (attachment.startsWith("{")) {
                res.put("details", JSONObject(attachment))
            } else {
                res.put("raw_attachment", attachment)
            }
        } catch (e: Exception) {
            res.put("error", e.message)
        }
        return res
    }

    /**
     * Get OpenChat URL
     */
    fun getOpenLink(chatId: Long): JSONObject {
        val sql = "SELECT link_id FROM chat_rooms WHERE id = ?"
        val rows = kakaoDB.executeQuery(sql, arrayOf(chatId.toString()))
        if (rows.isEmpty()) return JSONObject().put("error", "Room not found")
        val linkId = rows[0]["link_id"]
        val linkSql = "SELECT url FROM db2.open_link WHERE id = ?"
        val linkRows = kakaoDB.executeQuery(linkSql, arrayOf(linkId))
        val res = JSONObject()
        if (linkRows.isNotEmpty()) {
            res.put("url", linkRows[0]["url"])
            res.put("link_id", linkId)
        } else {
            res.put("error", "Link not found")
        }
        return res
    }

    /**
     * Get all tables
     */
    fun getTables(): JSONArray {
        val sql = "SELECT name FROM sqlite_master WHERE type='table' UNION SELECT name FROM db2.sqlite_master WHERE type='table'"
        val rows = kakaoDB.executeQuery(sql, null)
        val result = JSONArray()
        for (row in rows) {
            result.put(row["name"])
        }
        return result
    }

    /**
     * Get columns of a table
     */
    fun getColumns(table: String): JSONArray {
        val sql = "PRAGMA table_info($table)"
        val rows = kakaoDB.executeQuery(sql, null)
        val result = JSONArray()
        for (row in rows) {
            result.put(JSONObject(row))
        }
        return result
    }

    /**
     * Mark a room as read (DIRECT mode)
     */
    fun markAsReadDirect(chatId: Long): JSONObject {
        val intent = Intent().apply {
            component = ComponentName("com.kakao.talk", "com.kakao.talk.notification.NotificationActionService")
            putExtra("chat_id", chatId)
            action = "com.kakao.talk.notification.READ_MESSAGE"
        }
        return try {
            AndroidHiddenApi.startService(intent)
            JSONObject().put("success", true)
        } catch (e: Exception) {
            // Fallback to manual DB update
            try {
                val getLogSql = "SELECT last_log_id FROM chat_rooms WHERE id = ?"
                val logRows = kakaoDB.executeQuery(getLogSql, arrayOf(chatId.toString()))
                if (logRows.isNotEmpty()) {
                    val lastLogId = logRows[0]["last_log_id"]
                    val updateSql = "UPDATE chat_rooms SET last_read_log_id = ? WHERE id = ?"
                    kakaoDB.connection.execSQL(updateSql, arrayOf(lastLogId, chatId.toString()))
                    JSONObject().put("success", true).put("mode", "manual_db")
                } else {
                    JSONObject().put("success", false).put("error", "Room not found")
                }
            } catch (ex: Exception) {
                JSONObject().put("success", false).put("error", ex.message)
            }
        }
    }

    /**
     * Send message DIRECTLY (High performance)
     */
    fun sendMessageDirect(chatId: Long, msg: String): JSONObject {
        val intent = Intent().apply {
            component = ComponentName("com.kakao.talk", "com.kakao.talk.notification.NotificationActionService")
            putExtra("noti_referer", "stend")
            putExtra("chat_id", chatId)
            action = "com.kakao.talk.notification.REPLY_MESSAGE"

            val results = Bundle().apply {
                putCharSequence("reply_message", msg)
            }
            val remoteInput = RemoteInput.Builder("reply_message").build()
            RemoteInput.addResultsToIntent(arrayOf(remoteInput), this, results)
        }
        return try {
            AndroidHiddenApi.startService(intent)
            JSONObject().put("success", true)
        } catch (e: Exception) {
            JSONObject().put("success", false).put("error", e.message)
        }
    }

    /**
     * Get Auth Metadata (aot, d_id)
     */
    fun getAuthInfo(): JSONObject {
        val appPath = "/data/data/com.kakao.talk/"
        val prefsPath = "${appPath}shared_prefs/KakaoTalk.hw.perferences.xml"
        val aotPath = "${appPath}aot"

        return try {
            val prefs = File(prefsPath).readText()
            val dIdMatch = Regex("""<string name="d_id">\s*(.*?)\s*</string>""").find(prefs)
            val dId = dIdMatch?.groupValues?.get(1) ?: "Unknown"

            val aotEnc = File(aotPath).readText()
            
            val keys1 = intArrayOf(10, 2, 3, -4, 20, 73, 47, -38, 27, -22, 11, -20, -22, 37, 36, 54).map { it.toByte() }.toByteArray()
            val keys2 = intArrayOf(67, 109, -115, -110, -23, 119, 33, 86, -99, -28, -102, 109, -73, 13, 43, -96, 109, -76, 91, -83, 73, -14, 107, -88, 6, 11, 74, 109, 84, -68, -80, 15).map { it.toByte() }.toByteArray()
            
            val cipher = Cipher.getInstance("AES/CBC/PKCS5Padding")
            cipher.init(Cipher.DECRYPT_MODE, SecretKeySpec(keys2, "AES"), IvParameterSpec(keys1))
            
            val aotDec = String(cipher.doFinal(Base64.decode(aotEnc, Base64.DEFAULT)), StandardCharsets.UTF_8)
            JSONObject(aotDec).put("d_id", dId)
        } catch (e: Exception) {
            JSONObject().put("error", e.message)
        }
    }
}
