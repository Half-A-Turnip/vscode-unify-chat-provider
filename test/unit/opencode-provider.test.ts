import { beforeEach, describe, expect, it, vi } from 'vitest';
import type * as vscode from 'vscode';
import type { AuthTokenInfo } from '../../src/auth/types';
import type { RequestLogger } from '../../src/logger';
import type { ChatRequestTrace, ModelConfig, ProviderConfig } from '../../src/types';

const state = vi.hoisted(() => ({
  anthropicConfigs: [] as unknown[],
  anthropicStreamChat: vi.fn(),
  googleConfigs: [] as unknown[],
  googleStreamChat: vi.fn(),
  openAIChatConfigs: [] as unknown[],
  openAIChatEstimateTokenCount: vi.fn((_text: string) => 17),
  openAIChatGetAvailableModels: vi.fn<() => Promise<ModelConfig[]>>(),
  openAIChatStreamChat: vi.fn(),
  openAIResponsesConfigs: [] as unknown[],
  openAIResponsesStreamChat: vi.fn(),
}));

vi.mock('vscode', () => ({
  env: { language: 'en' },
  extensions: { getExtension: () => undefined },
  LanguageModelChatToolMode: { Auto: 1 },
  l10n: {
    t: (message: string | { message: string }) =>
      typeof message === 'string' ? message : message.message,
  },
}));

vi.mock('../../src/client/utils', () => ({
  matchProvider: (url: string, pattern: string | RegExp) =>
    typeof pattern === 'string'
      ? url.includes(pattern.replaceAll('*', ''))
      : pattern.test(url),
}));

vi.mock('../../src/client/anthropic/client', () => ({
  AnthropicProvider: class AnthropicProvider {
    constructor(config: unknown) {
      state.anthropicConfigs.push(config);
    }

    async *streamChat(...args: unknown[]): AsyncGenerator<never> {
      state.anthropicStreamChat(...args);
    }

    estimateTokenCount(): number {
      return 13;
    }
  },
}));

vi.mock('../../src/client/google/ai-studio-client', () => ({
  GoogleAIStudioProvider: class GoogleAIStudioProvider {
    constructor(config: unknown) {
      state.googleConfigs.push(config);
    }

    async *streamChat(...args: unknown[]): AsyncGenerator<never> {
      state.googleStreamChat(...args);
    }

    estimateTokenCount(): number {
      return 13;
    }
  },
}));

vi.mock('../../src/client/openai/chat-completion-client', () => ({
  OpenAIChatCompletionProvider: class OpenAIChatCompletionProvider {
    constructor(config: unknown) {
      state.openAIChatConfigs.push(config);
    }

    async *streamChat(...args: unknown[]): AsyncGenerator<never> {
      state.openAIChatStreamChat(...args);
    }

    estimateTokenCount(text: string): number {
      return state.openAIChatEstimateTokenCount(text);
    }

    async getAvailableModels(): Promise<ModelConfig[]> {
      return state.openAIChatGetAvailableModels();
    }
  },
}));

vi.mock('../../src/client/openai/responses-client', () => ({
  OpenAIResponsesProvider: class OpenAIResponsesProvider {
    constructor(config: unknown) {
      state.openAIResponsesConfigs.push(config);
    }

    async *streamChat(...args: unknown[]): AsyncGenerator<never> {
      state.openAIResponsesStreamChat(...args);
    }

    estimateTokenCount(): number {
      return 13;
    }
  },
}));

import { resolveEmptyChatResponsePolicy } from '../../src/client/interface';
import * as vscodeApi from 'vscode';
import {
  OPENCODE_GO_API_BASE_URL,
  OPENCODE_ZEN_API_BASE_URL,
  OpenCodeGoProvider,
  OpenCodeZenProvider,
  createOpenCodeDelegateConfigs,
  resolveOpenCodeProtocol,
  type OpenCodeService,
} from '../../src/client/opencode/provider';
import { WELL_KNOWN_PROVIDERS } from '../../src/well-known/providers';

function config(service: OpenCodeService): ProviderConfig {
  return {
    type: service === 'zen' ? 'opencode-zen' : 'opencode-go',
    name: service === 'zen' ? 'OpenCode Zen' : 'OpenCode Go',
    baseUrl: 'https://wrong.example.test/custom',
    useRawBaseUrl: true,
    transport: 'sse',
    serviceTier: 'priority',
    auth: { method: 'api-key', apiKey: 'secret-ref' },
    extraHeaders: { 'x-opencode-option': '1' },
    extraBody: { shared_provider_option: true },
    models: [],
    autoFetchOfficialModels: true,
  };
}

function trace(): ChatRequestTrace {
  return {
    performance: {
      tts: 0,
      ttf: 0,
      ttft: 0,
      tps: 0,
      tl: 0,
    },
  };
}

function cancellationToken(): vscode.CancellationToken {
  return {
    isCancellationRequested: false,
    onCancellationRequested: () => ({ dispose: () => undefined }),
  };
}

async function consume(source: AsyncIterable<unknown>): Promise<void> {
  for await (const part of source) {
    void part;
  }
}

beforeEach(() => {
  state.anthropicConfigs = [];
  state.anthropicStreamChat.mockReset();
  state.googleConfigs = [];
  state.googleStreamChat.mockReset();
  state.openAIChatConfigs = [];
  state.openAIChatEstimateTokenCount.mockClear();
  state.openAIChatGetAvailableModels.mockReset();
  state.openAIChatStreamChat.mockReset();
  state.openAIResponsesConfigs = [];
  state.openAIResponsesStreamChat.mockReset();
});

describe('OpenCode provider facade', () => {
  it('registers exactly one well-known provider per OpenCode service', () => {
    expect(
      WELL_KNOWN_PROVIDERS.filter((provider) =>
        provider.name.startsWith('OpenCode '),
      ),
    ).toEqual([
      expect.objectContaining({
        name: 'OpenCode Zen',
        type: 'opencode-zen',
        baseUrl: OPENCODE_ZEN_API_BASE_URL,
        authTypes: ['api-key'],
        models: [],
        autoFetchOfficialModels: true,
      }),
      expect.objectContaining({
        name: 'OpenCode Go',
        type: 'opencode-go',
        baseUrl: OPENCODE_GO_API_BASE_URL,
        authTypes: ['api-key'],
        models: [],
        autoFetchOfficialModels: true,
      }),
    ]);
  });

  it.each([
    ['zen', OPENCODE_ZEN_API_BASE_URL],
    ['go', OPENCODE_GO_API_BASE_URL],
  ] as const)(
    'constructs every %s protocol delegate against the canonical API base URL',
    (service, expectedBaseUrl) => {
      const providerConfig = config(service);
      const delegates = createOpenCodeDelegateConfigs(providerConfig, service);

      expect(delegates.openAIChat).toMatchObject({
        type: 'openai-chat-completion',
        baseUrl: expectedBaseUrl,
        useRawBaseUrl: false,
        transport: 'sse',
        serviceTier: 'priority',
        extraBody: { shared_provider_option: true },
      });
      expect(delegates.openAIResponses.type).toBe('openai-responses');
      expect(delegates.anthropic.type).toBe('anthropic');
      expect(delegates.google.type).toBe('google-ai-studio');
      for (const delegate of Object.values(delegates)) {
        expect(delegate.baseUrl).toBe(expectedBaseUrl);
        expect(delegate.auth).toBe(providerConfig.auth);
        expect(delegate.extraHeaders).toBe(providerConfig.extraHeaders);
        expect(delegate.models).toBe(providerConfig.models);
      }

      if (service === 'zen') new OpenCodeZenProvider(providerConfig);
      else new OpenCodeGoProvider(providerConfig);

      expect(state.openAIChatConfigs).toEqual([delegates.openAIChat]);
      expect(state.openAIResponsesConfigs).toEqual([
        delegates.openAIResponses,
      ]);
      expect(state.anthropicConfigs).toEqual([delegates.anthropic]);
      expect(state.googleConfigs).toEqual([delegates.google]);
    },
  );

  it.each([
    ['zen', 'gpt-5.6-sol', 'openai-responses'],
    ['zen', 'grok-4.6', 'openai-responses'],
    ['zen', 'muse-spark-1.2', 'openai-responses'],
    ['zen', 'claude-sonnet-5', 'anthropic-messages'],
    ['zen', 'qwen3.6-plus', 'anthropic-messages'],
    ['zen', 'GEMINI-3.7-FLASH#thinking', 'google-generative-ai'],
    ['zen', 'minimax-m3-free', 'anthropic-messages'],
    ['zen', 'minimax-m3', 'openai-chat-completions'],
    ['zen', 'deepseek-v4-flash', 'openai-chat-completions'],
    ['go', 'gpt-5.6-luna', 'openai-responses'],
    ['go', 'grok-4.6', 'openai-responses'],
    ['go', 'muse-spark-1.2-contributor', 'openai-responses'],
    ['go', 'minimax-m3', 'anthropic-messages'],
    ['go', 'qwen3.8-max', 'anthropic-messages'],
    ['go', 'gemini-future', 'openai-chat-completions'],
    ['go', 'future-model', 'openai-chat-completions'],
  ] as const)(
    'routes %s model %s through %s',
    (service, id, expectedProtocol) => {
      expect(resolveOpenCodeProtocol(service, { id })).toBe(expectedProtocol);
    },
  );

  it('forwards every stream argument to the selected protocol delegate', async () => {
    const provider = new OpenCodeZenProvider(config('zen'));
    const messages: readonly vscode.LanguageModelChatRequestMessage[] = [];
    const options: vscode.ProvideLanguageModelChatResponseOptions = {
      requestInitiator: 'test',
      toolMode: vscodeApi.LanguageModelChatToolMode.Auto,
      tools: [],
    };
    const requestTrace = trace();
    const token = cancellationToken();
    const logger = Object.create(null) as RequestLogger;
    const credential: AuthTokenInfo = {
      kind: 'token',
      token: 'zen-key',
      tokenType: 'Bearer',
    };
    const refreshCredential = vi.fn(async () => credential);

    const routes = [
      ['claude-sonnet-5', state.anthropicStreamChat],
      ['gemini-3.7-flash', state.googleStreamChat],
      ['deepseek-v4-flash', state.openAIChatStreamChat],
      ['gpt-5.6-sol', state.openAIResponsesStreamChat],
    ] as const;

    for (const [id, stream] of routes) {
      const model: ModelConfig = { id, extraBody: { model_option: id } };
      await consume(
        provider.streamChat(
          `OpenCode Zen/${id}`,
          model,
          messages,
          options,
          requestTrace,
          token,
          logger,
          credential,
          refreshCredential,
        ),
      );
      expect(stream).toHaveBeenCalledWith(
        `OpenCode Zen/${id}`,
        model,
        messages,
        options,
        requestTrace,
        token,
        logger,
        credential,
        refreshCredential,
      );
    }

    expect(state.anthropicStreamChat).toHaveBeenCalledOnce();
    expect(state.googleStreamChat).toHaveBeenCalledOnce();
    expect(state.openAIChatStreamChat).toHaveBeenCalledOnce();
    expect(state.openAIResponsesStreamChat).toHaveBeenCalledOnce();
  });

  it('uses the Messages empty-response policy only for routed Messages models', () => {
    const zen = new OpenCodeZenProvider(config('zen'));
    const go = new OpenCodeGoProvider(config('go'));

    expect(
      resolveEmptyChatResponsePolicy({}, zen, { id: 'claude-sonnet-5' }),
    ).toBe('success');
    expect(
      resolveEmptyChatResponsePolicy({}, zen, { id: 'gpt-5.6-sol' }),
    ).toBe('retry');
    expect(
      resolveEmptyChatResponsePolicy({}, go, { id: 'minimax-m3' }),
    ).toBe('success');
  });

  it('fetches one combined model catalog through the Chat Completions delegate', async () => {
    const catalogModels: ModelConfig[] = [
      { id: 'claude-sonnet-5' },
      { id: 'gemini-3.7-flash' },
      { id: 'gpt-5.6-sol' },
      { id: 'deepseek-v4-flash' },
      { id: 'deepseek-v4-flash-free' },
    ];
    state.openAIChatGetAvailableModels.mockResolvedValue(catalogModels);
    const provider = new OpenCodeZenProvider(config('zen'));
    const credential: AuthTokenInfo = {
      kind: 'token',
      token: 'zen-key',
      tokenType: 'Bearer',
    };
    const refreshCredential = vi.fn(async () => credential);
    const controller = new AbortController();

    await expect(
      provider.getAvailableModels(
        credential,
        refreshCredential,
        controller.signal,
      ),
    ).resolves.toEqual([
      ...catalogModels.slice(0, -1),
      {
        id: 'deepseek-v4-flash-free',
        name: 'DeepSeek V4 Flash (Free)',
      },
    ]);
    expect(state.openAIChatGetAvailableModels).toHaveBeenCalledOnce();
    expect(provider.estimateTokenCount('count me')).toBe(17);
    expect(state.openAIChatEstimateTokenCount).toHaveBeenCalledWith('count me');
  });
});
