package party.qwer.iris

import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.KotlinxWebsocketSerializationConverter
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.statuspages.StatusPages
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import io.ktor.server.websocket.WebSockets
import io.ktor.server.websocket.webSocket
import io.ktor.websocket.send
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.jsonObject
import party.qwer.iris.model.AotResponse
import party.qwer.iris.model.ApiResponse
import party.qwer.iris.model.CommonErrorResponse
import party.qwer.iris.model.ConfigRequest
import party.qwer.iris.model.ConfigResponse
import party.qwer.iris.model.DashboardStatusResponse
import party.qwer.iris.model.DecryptRequest
import party.qwer.iris.model.DecryptResponse
import party.qwer.iris.model.QueryRequest
import party.qwer.iris.model.QueryResponse
import party.qwer.iris.model.ReplyRequest
import party.qwer.iris.model.ReplyType


class IrisServer(
    private val kakaoDB: KakaoDB,
    private val dbObserver: DBObserver,
    private val observerHelper: ObserverHelper,
    private val notificationReferer: String,
    private val wsBroadcastFlow: MutableSharedFlow<String>
) {
    val sharedFlow = wsBroadcastFlow.asSharedFlow()

    fun startServer() {
        embeddedServer(Netty, port = Configurable.botSocketPort) {
            install(WebSockets) {
                contentConverter = KotlinxWebsocketSerializationConverter(Json)
            }

            install(ContentNegotiation) {
                json()
            }

            install(StatusPages) {
                exception<Throwable> { call, cause ->
                    call.respond(
                        HttpStatusCode.InternalServerError, CommonErrorResponse(
                            message = cause.message ?: "unknown error"
                        )
                    )
                }
            }

            routing {
                route("/dashboard") {
                    get {
                        val html = PageRenderer.renderDashboard()
                        call.respondText(html, ContentType.Text.Html)
                    }

                    get("status") {
                        call.respond(
                            DashboardStatusResponse(
                                isObserving = dbObserver.isPollingThreadAlive,
                                statusMessage = if (dbObserver.isPollingThreadAlive) {
                                    "Observing database"
                                } else {
                                    "Not observing database"
                                },
                                lastLogs = observerHelper.lastChatLogs
                            )
                        )
                    }
                }

                route("/config") {
                    get {
                        call.respond(
                            ConfigResponse(
                                bot_name = Configurable.botName,
                                bot_http_port = Configurable.botSocketPort,
                                web_server_endpoint = Configurable.webServerEndpoint,
                                db_polling_rate = Configurable.dbPollingRate,
                                message_send_rate = Configurable.messageSendRate,
                                bot_id = Configurable.botId,
                            )
                        )
                    }

                    post("{name}") {
                        val name = call.parameters["name"]
                        val req = call.receive<ConfigRequest>()

                        when (name) {
                            "endpoint" -> {
                                var value = req.endpoint
                                if (value == null) {
                                    value = ""
                                }
                                Configurable.webServerEndpoint = value
                            }

                            "botname" -> {
                                val value = req.botname
                                if (value.isNullOrBlank()) {
                                    throw Exception("missing or empty value")
                                }
                                Configurable.botName = value
                            }

                            "dbrate" -> {
                                val value = req.rate ?: throw Exception("missing or invalid value")

                                Configurable.dbPollingRate = value
                            }

                            "sendrate" -> {
                                val value = req.rate ?: throw Exception("missing or invalid value")

                                Configurable.messageSendRate = value
                            }

                            "botport" -> {
                                val value = req.port ?: throw Exception("missing or invalid value")

                                if (value < 1 || value > 65535) {
                                    throw Exception("Invalid port number. Port must be between 1 and 65535.")
                                }

                                Configurable.botSocketPort = value
                            }

                            else -> {
                                throw Exception("Unknown config $name")
                            }
                        }

                        call.respond(ApiResponse(success = true, message = "success"))
                    }
                }

                get("/aot") {
                    val aotToken = AuthProvider.getToken()

                    call.respond(
                        AotResponse(
                            success = true,
                            aot = Json.parseToJsonElement(aotToken.toString()).jsonObject
                        )
                    )
                }

                post("/reply") {
                    val replyRequest = call.receive<ReplyRequest>()
                    val roomId = replyRequest.room.toLong()
                    val threadId = replyRequest.threadId?.toLong()

                    when (replyRequest.type) {
                        ReplyType.TEXT -> Replier.sendMessage(
                            notificationReferer,
                            roomId,
                            replyRequest.data.jsonPrimitive.content,
                            threadId
                        )

                        ReplyType.IMAGE -> Replier.sendPhoto(
                            roomId, replyRequest.data.jsonPrimitive.content
                        )

                        ReplyType.IMAGE_MULTIPLE -> Replier.sendMultiplePhotos(
                            roomId,
                            replyRequest.data.jsonArray.map { it.jsonPrimitive.content })
                    }

                    call.respond(ApiResponse(success = true, message = "success"))
                }

                post("/query") {
                    val queryRequest = call.receive<QueryRequest>()

                    try {
                        val rows = kakaoDB.executeQuery(
                            queryRequest.query,
                            (queryRequest.bind?.map { it.content } ?: listOf()).toTypedArray())

                        call.respond(QueryResponse(data = rows.map {
                            KakaoDB.decryptRow(it)
                        }))
                    } catch (e: Exception) {
                        throw Exception("Query 오류: query=${queryRequest.query}, err=${e.message}")
                    }
                }

                post("/decrypt") {
                    val decryptRequest = call.receive<DecryptRequest>()
                    val plaintext = KakaoDecrypt.decrypt(
                        decryptRequest.enc,
                        decryptRequest.b64_ciphertext,
                        decryptRequest.user_id ?: Configurable.botId
                    )

                    call.respond(DecryptResponse(plain_text = plaintext))
                }

                val grandService = StendGrandService(kakaoDB)

                route("/api/v1") {
                    get("/rooms") {
                        call.respondText(grandService.getRoomList().toString(), ContentType.Application.Json)
                    }
                    get("/rooms/{id}/members") {
                        val id = call.parameters["id"]?.toLong() ?: 0L
                        call.respondText(grandService.getMembers(id).toString(), ContentType.Application.Json)
                    }
                    get("/rooms/{id}/history") {
                        val id = call.parameters["id"]?.toLong() ?: 0L
                        val limit = call.request.queryParameters["limit"]?.toInt() ?: 100
                        call.respondText(grandService.getHistory(id, limit).toString(), ContentType.Application.Json)
                    }
                    get("/rooms/{id}/search") {
                        val id = call.parameters["id"]?.toLong() ?: 0L
                        val query = call.request.queryParameters["q"] ?: ""
                        val limit = call.request.queryParameters["limit"]?.toInt() ?: 100
                        call.respondText(grandService.search(id, query, limit).toString(), ContentType.Application.Json)
                    }
                    get("/rooms/{id}/link") {
                    val id = call.parameters["id"]?.toLongOrNull() ?: 0L
                    call.respond(grandService.getOpenLink(id).toString())
                }
                
                // --- Power Features ---
                post("/chats/{id}/send_direct") {
                    val id = call.parameters["id"]?.toLongOrNull() ?: 0L
                    val msg = call.receiveText() // Expecting raw message string
                    call.respond(grandService.sendMessageDirect(id, msg).toString())
                }
                
                post("/rooms/{id}/read_direct") {
                    val id = call.parameters["id"]?.toLongOrNull() ?: 0L
                    call.respond(grandService.markAsReadDirect(id).toString())
                }
                
                get("/auth/info") {
                    call.respond(grandService.getAuthInfo().toString())
                }
                    get("/db/tables") {
                        call.respondText(grandService.getTables().toString(), ContentType.Application.Json)
                    }
                    get("/db/columns") {
                        val table = call.request.queryParameters["table"] ?: ""
                        call.respondText(grandService.getColumns(table).toString(), ContentType.Application.Json)
                    }
                    get("/chats/{id}/media_info") {
                        val id = call.parameters["id"]?.toLong() ?: 0L
                        call.respondText(grandService.getMediaInfo(id).toString(), ContentType.Application.Json)
                    }
                    get("/chats/{id}/context") {
                        val id = call.parameters["id"]?.toLong() ?: 0L
                        val limit = call.request.queryParameters["limit"]?.toInt() ?: 10
                        val dir = call.request.queryParameters["dir"] ?: "prev"
                        call.respondText(grandService.getContext(id, limit, dir).toString(), ContentType.Application.Json)
                    }
                    get("/rooms/{id}/stats") {
                        val id = call.parameters["id"]?.toLong() ?: 0L
                        call.respondText(grandService.getStats(id).toString(), ContentType.Application.Json)
                    }
                    post("/rooms/{id}/read") {
                        val id = call.parameters["id"]?.toLong() ?: 0L
                        call.respondText(grandService.markAsRead(id).toString(), ContentType.Application.Json)
                    }
                    get("/users/{id}") {
                        val id = call.parameters["id"]?.toLong() ?: 0L
                        call.respondText(grandService.getUserInfo(id).toString(), ContentType.Application.Json)
                    }
                    get("/friends") {
                        call.respondText(grandService.getFriends().toString(), ContentType.Application.Json)
                    }
                    // Proxies for legacy/core
                    post("/query") {
                        val queryRequest = call.receive<QueryRequest>()
                        val rows = kakaoDB.executeQuery(queryRequest.query, (queryRequest.bind?.map { it.content } ?: listOf()).toTypedArray())
                        call.respond(QueryResponse(data = rows.map { KakaoDB.decryptRow(it) }))
                    }
                    post("/decrypt") {
                        val decryptRequest = call.receive<DecryptRequest>()
                        val plaintext = KakaoDecrypt.decrypt(decryptRequest.enc, decryptRequest.b64_ciphertext, decryptRequest.user_id ?: Configurable.botId)
                        call.respond(DecryptResponse(plain_text = plaintext))
                    }
                }

                webSocket("/ws") {
                    sharedFlow.collect { msg ->
                        send(msg)
                    }
                }
            }
        }.start(wait = true)
    }
}
