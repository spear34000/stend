package party.qwer.iris

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Bundle
import android.os.IBinder

@SuppressLint("PrivateApi")
class AndroidHiddenApi {
    companion object {
        val startService = getStartServiceMethod()
        val startActivity = getStartActivityMethod()
        val broadcastIntent = getBroadcastIntentMethod()

        private val callingPackageName: String by lazy {
            "com.android.shell"
        }

        private fun getStartServiceMethod(): (Intent) -> Unit {
            val IActivityManagerStub = Class.forName("android.app.IActivityManager\$Stub")
            val IActivityManager = Class.forName("android.app.IActivityManager")
            val IApplicationThread = Class.forName("android.app.IApplicationThread")

            val activityManager =
                IActivityManagerStub.getMethod("asInterface", IBinder::class.java).invoke(
                    null, getService("activity")
                )

            try {
                // Latest Android versions
                val method = IActivityManager.getMethod(
                    "startService",
                    IApplicationThread,
                    Intent::class.java,
                    java.lang.String::class.java,
                    java.lang.Boolean.TYPE,
                    java.lang.String::class.java,
                    java.lang.String::class.java,
                    java.lang.Integer.TYPE,
                )

                return { intent ->
                    method.invoke(
                        activityManager, null, intent, null, false, callingPackageName, null, -3
                    )
                }
            } catch (_: Exception) {}

            try {
                // Older Android versions
                val method = IActivityManager.getMethod(
                    "startService",
                    IApplicationThread,
                    Intent::class.java,
                    java.lang.String::class.java,
                    java.lang.Boolean.TYPE,
                    java.lang.String::class.java,
                    java.lang.Integer.TYPE,
                )

                return { intent ->
                    method.invoke(
                        activityManager, null, intent, null, false, callingPackageName, -3
                    )
                }
            } catch (_: Exception) {}

            throw Exception("Failed to get startService Method")
        }

        private fun getStartActivityMethod(): (Intent) -> Unit {
            val IActivityManagerStub = Class.forName("android.app.IActivityManager\$Stub")
            val IActivityManager = Class.forName("android.app.IActivityManager")
            val IApplicationThread = Class.forName("android.app.IApplicationThread")

            val activityManager =
                IActivityManagerStub.getMethod("asInterface", IBinder::class.java).invoke(
                    null, getService("activity")
                )

            try {
                val ProfilerInfo = Class.forName("android.app.ProfilerInfo")
                val method = IActivityManager.getMethod(
                    "startActivity",
                    IApplicationThread,
                    String::class.java,
                    String::class.java,
                    Intent::class.java,
                    String::class.java,
                    IBinder::class.java,
                    String::class.java,
                    Integer.TYPE,
                    Integer.TYPE,
                    ProfilerInfo,
                    Bundle::class.java,
                    Integer.TYPE
                )

                return { intent ->
                    method.invoke(
                        activityManager,
                        null,
                        callingPackageName,
                        null,
                        intent,
                        intent.type,
                        null,
                        null,
                        0,
                        0,
                        null,
                        null,
                        -3
                    )
                }
            } catch (_: Exception) {}
            
            throw Exception("Failed to get startActivity Method")
        }

        private fun getBroadcastIntentMethod(): (Intent) -> Unit {
            val IActivityManagerStub = Class.forName("android.app.IActivityManager\$Stub")
            val IActivityManager = Class.forName("android.app.IActivityManager")
            val IApplicationThread = Class.forName("android.app.IApplicationThread")

            val activityManager =
                IActivityManagerStub.getMethod("asInterface", IBinder::class.java).invoke(
                    null, getService("activity")
                )

            try {
                val IIntentReceiver = Class.forName("android.content.IIntentReceiver")
                val method = IActivityManager.getMethod(
                    "broadcastIntent",
                    IApplicationThread,
                    Intent::class.java,
                    String::class.java,
                    IIntentReceiver,
                    Integer.TYPE,
                    String::class.java,
                    Bundle::class.java,
                    Array<String>::class.java,
                    Integer.TYPE,
                    Bundle::class.java,
                    Boolean::class.java,
                    Boolean::class.java,
                    Int::class.java
                )

                return { intent ->
                    method.invoke(
                        activityManager,
                        null,
                        intent,
                        null,
                        null,
                        0,
                        null,
                        null,
                        null,
                        -1,
                        null,
                        false,
                        false,
                        -3
                    )
                }
            } catch (_: Exception) {}
            
            throw Exception("Failed to get broadcastIntent Method")
        }

        private fun getService(name: String): IBinder {
            val method = Class.forName("android.os.ServiceManager")
                .getMethod("getService", String::class.java)

            return method.invoke(null, name) as IBinder
        }
    }
}