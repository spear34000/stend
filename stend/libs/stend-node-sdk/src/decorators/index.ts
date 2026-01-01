import 'reflect-metadata';

/**
 * @file index.ts
 * @description Stend Node SDK Decorators
 * @reference Derived from Iris Platform architectural patterns.
 */

export const STEND_COMMAND_KEY = 'stend:command';
export const STEND_EVENT_KEY = 'stend:event';
export const STEND_PREFIX_KEY = 'stend:prefix';

export interface CommandOptions {
    trigger: string | string[];
    description?: string;
    usage?: string;
    aliases?: string[];
}

export interface EventOptions {
    type: 'message' | 'event' | 'join' | 'leave' | 'group_event';
    filters?: string[];
}

/**
 * Registers a method as a bot command.
 */
export function Command(options: CommandOptions): MethodDecorator {
    return (target: any, propertyKey: string | symbol, descriptor: PropertyDescriptor) => {
        Reflect.defineMetadata(STEND_COMMAND_KEY, options, descriptor.value);
    };
}

/**
 * Registers a method to handle specific Stend events.
 */
export function EventHandler(options: EventOptions): MethodDecorator {
    return (target: any, propertyKey: string | symbol, descriptor: PropertyDescriptor) => {
        Reflect.defineMetadata(STEND_EVENT_KEY, options, descriptor.value);
    };
}

/**
 * Sets a global prefix for commands in a class.
 */
export function UsePrefix(prefix: string): ClassDecorator {
    return (target: Function) => {
        Reflect.defineMetadata(STEND_PREFIX_KEY, prefix, target);
    };
}
