import WebSocket from 'ws';
import axios from 'axios';
import { STEND_COMMAND_KEY, STEND_EVENT_KEY, STEND_PREFIX_KEY } from '../decorators';

/**
 * @interface StendOptions
 * @description Configuration for the Stend Bot instance.
 */
export interface StendOptions {
    name: string;
    endpoint: string;
}

/**
 * @class StendBot
 * @description Core bot controller for the Stend Platform.
 * @reference Architecturally inspired by Iris core dispatchers.
 */
export class StendBot {
    private socket: WebSocket | null = null;
    private eventHandlers: Map<string, Function[]> = new Map();
    private commandRegistry: Map<string, { handler: Function, meta: any }> = new Map();
    private defaultPrefix: string = '/';

    constructor(protected readonly options: StendOptions) {
        this.initializeMetadata();
    }

    /**
     * Reflects class metadata to register decorated commands and events.
     */
    private initializeMetadata() {
        const proto = Object.getPrototypeOf(this);
        const members = Object.getOwnPropertyNames(proto);
        
        // Handle class-level prefix
        const classPrefix = Reflect.getMetadata(STEND_PREFIX_KEY, this.constructor);
        if (classPrefix) this.defaultPrefix = classPrefix;

        for (const key of members) {
            const member = (this as any)[key];
            if (typeof member !== 'function' || key === 'constructor') continue;

            // Register Decorated Commands
            const cmdMeta = Reflect.getMetadata(STEND_COMMAND_KEY, member);
            if (cmdMeta) {
                const triggers = Array.isArray(cmdMeta.trigger) ? cmdMeta.trigger : [cmdMeta.trigger];
                for (const t of triggers) {
                    this.commandRegistry.set(t, { handler: member.bind(this), meta: cmdMeta });
                }
            }

            // Register Decorated Event Handlers
            const eventMeta = Reflect.getMetadata(STEND_EVENT_KEY, member);
            if (eventMeta) {
                const type = eventMeta.type;
                if (!this.eventHandlers.has(type)) this.eventHandlers.set(type, []);
                this.eventHandlers.get(type)?.push(member.bind(this));
            }
        }
    }

    /**
     * Manual event registration.
     */
    public on(event: string, handler: Function) {
        if (!this.eventHandlers.has(event)) this.eventHandlers.set(event, []);
        this.eventHandlers.get(event)?.push(handler);
    }

    /**
     * Connects to the Stend API and begins listening for events.
     */
    public async start() {
        const wsUrl = this.options.endpoint.replace('http', 'ws') + '/ws';
        this.socket = new WebSocket(wsUrl);

        this.socket.on('open', () => {
            console.log(`[Stend] Bot '${this.options.name}' online @ ${wsUrl}`);
        });

        this.socket.on('message', async (data: string) => {
            try {
                const packet = JSON.parse(data);
                await this.processPacket(packet);
            } catch (err) {
                console.error('[Stend] Failed to process incoming packet:', err);
            }
        });

        this.socket.on('close', () => {
            console.log('[Stend] Disconnected. Re-establishing link in 5s...');
            setTimeout(() => this.start(), 5000);
        });
    }

    private async processPacket(packet: any) {
        const ctx = this.createContext(packet);

        // Core message dispatch
        const handlers = this.eventHandlers.get('message') || [];
        for (const h of handlers) await h(ctx);

        // Command processing
        const content = ctx.message.text.trim();
        if (content.startsWith(this.defaultPrefix)) {
            const tokens = content.slice(this.defaultPrefix.length).split(/\s+/);
            const cmdName = tokens[0];
            const args = tokens.slice(1);

            const cmd = this.commandRegistry.get(cmdName);
            if (cmd) {
                await cmd.handler(ctx, args);
            }
        }
    }

    /**
     * Maps raw Stend packets to a developer-friendly context.
     */
    private createContext(packet: any) {
        const rawJson = packet.json || {};
        return {
            room: { id: rawJson.chat_id, name: packet.room, type: packet.type },
            sender: { id: rawJson.user_id, name: packet.sender },
            message: { text: packet.msg, id: rawJson.id, timestamp: packet.created_at },
            reply: async (msg: string) => {
                await axios.post(`${this.options.endpoint}/api/stend/reply`, {
                    room: rawJson.chat_id,
                    data: msg,
                    type: 'text'
                });
            },
            raw: packet
        };
    }
}
