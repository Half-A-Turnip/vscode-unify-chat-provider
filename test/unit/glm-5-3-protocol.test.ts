import { describe, expect, it, vi } from 'vitest';

vi.mock('vscode', () => {
  class EventEmitter<T> {
    readonly event = (
      _listener: (value: T) => unknown,
    ): { dispose: () => void } => ({ dispose: () => undefined });
    fire(_value: T): void {}
    dispose(): void {}
  }

  class ThemeIcon {
    constructor(readonly id: string) {}
  }

  return {
    env: { language: 'en' },
    EventEmitter,
    extensions: { getExtension: () => undefined },
    l10n: {
      t: (message: string | { message: string }) =>
        typeof message === 'string' ? message : message.message,
    },
    ThemeIcon,
    workspace: {
      getConfiguration: () => ({
        get: (_key: string, fallback?: unknown) => fallback,
      }),
    },
  };
});

import { FeatureId } from '../../src/client/definitions';
import { OpenAIChatCompletionProvider } from '../../src/client/openai/chat-completion-client';
import { isFeatureSupported } from '../../src/client/utils';
import type { ModelConfig, ProviderConfig } from '../../src/types';

const OFFICIAL_ENDPOINTS = [
  'https://open.bigmodel.cn/api/paas/v4',
  'https://open.bigmodel.cn/api/coding/paas/v4',
  'https://api.z.ai/api/paas/v4',
  'https://api.z.ai/api/coding/paas/v4',
] as const;

const SUPPORTED_IDENTITIES = [
  { id: 'glm-5.3' },
  { id: 'glm-5.3-flash' },
  { id: 'z-ai/glm-5.3' },
  { id: 'z-ai/glm-5.3-flash' },
  { id: 'custom-glm', family: 'glm-5.3' },
  { id: 'custom-glm-flash', family: 'glm-5.3-flash' },
  { id: 'custom-z-ai-glm', family: 'z-ai/glm-5.3' },
  { id: 'custom-z-ai-glm-flash', family: 'z-ai/glm-5.3-flash' },
] satisfies ReadonlyArray<{ id: string; family?: string }>;

const FORCED_REASONING_TYPE = 'forced_thinking_with_reasoning_effort';

function providerConfig(
  baseUrl: string,
  type: ProviderConfig['type'] = 'openai-chat-completion',
): ProviderConfig {
  return {
    type,
    name: 'GLM-5.3 protocol test',
    baseUrl,
    models: [],
  };
}

function modelConfig(
  identity: { id: string; family?: string } = { id: 'glm-5.3' },
  thinking: NonNullable<ModelConfig['thinking']> = {
    type: 'enabled',
    effort: 'max',
  },
): ModelConfig {
  return { ...identity, thinking };
}

const client = new OpenAIChatCompletionProvider(
  providerConfig(OFFICIAL_ENDPOINTS[0]),
);

function invokeReasoningBuilder(
  methodName: 'buildReasoningParams' | 'buildReasoningExtraBody',
  model: ModelConfig,
): unknown {
  const builder: unknown = Reflect.get(client, methodName);
  if (typeof builder !== 'function') {
    throw new Error(`${methodName} was not found`);
  }
  return Reflect.apply(builder, client, [model, FORCED_REASONING_TYPE]);
}

describe('GLM-5.3 protocol feature boundaries', () => {
  it.each(OFFICIAL_ENDPOINTS)(
    'enables the feature for every supported identity on %s',
    (baseUrl) => {
      for (const identity of SUPPORTED_IDENTITIES) {
        expect(
          isFeatureSupported(
            FeatureId.OpenAIUseGlm53ReasoningEffortParam,
            providerConfig(baseUrl),
            modelConfig(identity),
          ),
          `${identity.id} / ${identity.family ?? '(no family)'}`,
        ).toBe(true);
      }
    },
  );

  it.each([
    { id: 'glm-5.2' },
    { id: 'glm-5.30' },
    { id: 'glm-5.3-pro' },
    { id: 'glm-5.3-flash-pro' },
    { id: 'vendor/glm-5.3' },
    { id: 'hf:zai-org/GLM-5.3-Flash' },
    { id: 'custom-glm', family: 'glm-5' },
    { id: 'custom-glm', family: 'z-ai/glm-5.30' },
  ] satisfies ReadonlyArray<{ id: string; family?: string }>)(
    'rejects unsupported identity $id / $family',
    (identity) => {
      expect(
        isFeatureSupported(
          FeatureId.OpenAIUseGlm53ReasoningEffortParam,
          providerConfig(OFFICIAL_ENDPOINTS[0]),
          modelConfig(identity),
        ),
      ).toBe(false);
    },
  );

  it.each(OFFICIAL_ENDPOINTS)(
    'rejects non-Chat Completion protocol on %s',
    (baseUrl) => {
      expect(
        isFeatureSupported(
          FeatureId.OpenAIUseGlm53ReasoningEffortParam,
          providerConfig(baseUrl, 'openai-responses'),
          modelConfig(),
        ),
      ).toBe(false);
    },
  );

  it.each(OFFICIAL_ENDPOINTS)(
    'accepts a trailing slash on official endpoint %s',
    (baseUrl) => {
      expect(
        isFeatureSupported(
          FeatureId.OpenAIUseGlm53ReasoningEffortParam,
          providerConfig(`${baseUrl}/`),
          modelConfig(),
        ),
      ).toBe(true);
    },
  );

  it.each([
    'http://api.z.ai/api/paas/v4',
    'https://api.z.ai:8443/api/paas/v4',
    'https://api.z.ai/api/paas',
    'https://api.z.ai/api/paas/v4/chat/completions',
    'https://api.z.ai.example/api/paas/v4',
    'https://user@api.z.ai/api/paas/v4',
    'https://api.z.ai/api/paas/v4?route=other',
    'https://api.z.ai/api/paas/v4#fragment',
  ])('rejects non-exact endpoint %s', (baseUrl) => {
    expect(
      isFeatureSupported(
        FeatureId.OpenAIUseGlm53ReasoningEffortParam,
        providerConfig(baseUrl),
        modelConfig(),
      ),
    ).toBe(false);
  });
});

describe('GLM-5.3 forced reasoning parameters', () => {
  it.each([
    { type: 'enabled' as const, effort: 'low' as const },
    { type: 'auto' as const, effort: 'max' as const },
    { type: 'disabled' as const },
    { type: 'enabled' as const, effort: 'none' as const },
  ])('always sends thinking.enabled for $type / $effort', (thinking) => {
    expect(
      invokeReasoningBuilder(
        'buildReasoningParams',
        modelConfig(undefined, thinking),
      ),
    ).toEqual({ thinking: { type: 'enabled' } });
  });

  it.each(['low', 'high', 'max'] as const)(
    'sends reasoning_effort=%s unchanged',
    (effort) => {
      expect(
        invokeReasoningBuilder(
          'buildReasoningExtraBody',
          modelConfig(undefined, { type: 'enabled', effort }),
        ),
      ).toEqual({ reasoning_effort: effort });
    },
  );

  it.each([
    { type: 'disabled' as const },
    { type: 'enabled' as const, effort: 'none' as const },
  ])('downgrades stale $type / $effort to low', (thinking) => {
    expect(
      invokeReasoningBuilder(
        'buildReasoningExtraBody',
        modelConfig(undefined, thinking),
      ),
    ).toEqual({ reasoning_effort: 'low' });
  });
});
