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
import { isFeatureSupported } from '../../src/client/utils';
import { applyPresetTemplateSelections } from '../../src/preset-templates';
import type { ModelConfig, ProviderConfig } from '../../src/types';
import {
  getAlternativeIds,
  WELL_KNOWN_MODELS,
} from '../../src/well-known/models';
import {
  resolveProviderModels,
  WELL_KNOWN_PROVIDERS,
} from '../../src/well-known/providers';

const PAYG_MODELS = [
  'qwen3.8-max',
  'qwen3.8-flash',
  'qwen3.8-27b',
  'qwen3.7-max',
  'qwen3.7-plus',
  'qwen3.6-flash',
] as const;

const INDIVIDUAL_MODELS = [
  'qwen3.8-max',
  'qwen3.8-flash',
  'qwen3.7-max',
  'qwen3.7-plus',
  'qwen3.6-flash',
  'glm-5.2',
  'deepseek-v4-pro',
] as const;

const TEAM_MODELS = [
  'qwen3.8-max',
  'qwen3.8-flash',
  'qwen3.7-max',
  'qwen3.7-plus',
  'qwen3.6-plus',
  'qwen3.6-flash',
  'deepseek-v4-pro',
  'deepseek-v4-flash',
  'deepseek-v3.2',
  'kimi-k2.7-code',
  'kimi-k2.6',
  'kimi-k2.5',
  'glm-5.2',
  'glm-5.1',
  'glm-5',
  'MiniMax-M2.5',
] as const;

const QWEN_CLOUD_PROVIDERS = [
  {
    name: 'Qwen Cloud (China)',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    models: PAYG_MODELS,
    payAsYouGo: true,
  },
  {
    name: 'Qwen Cloud (China, Token Plan Individual)',
    baseUrl:
      'https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
    models: INDIVIDUAL_MODELS,
    payAsYouGo: false,
  },
  {
    name: 'Qwen Cloud (China, Token Plan Team Edition)',
    baseUrl:
      'https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
    models: TEAM_MODELS,
    payAsYouGo: false,
  },
  {
    name: 'Qwen Cloud (International)',
    baseUrl: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    models: PAYG_MODELS,
    payAsYouGo: true,
  },
  {
    name: 'Qwen Cloud (International, Token Plan Individual)',
    baseUrl:
      'https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1',
    models: INDIVIDUAL_MODELS,
    payAsYouGo: false,
  },
  {
    name: 'Qwen Cloud (International, Token Plan Team Edition)',
    baseUrl:
      'https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1',
    models: TEAM_MODELS,
    payAsYouGo: false,
  },
] as const;

const OFFICIAL_QWEN_ENDPOINTS = [
  'https://dashscope.aliyuncs.com/compatible-mode/v1',
  'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
  'https://dashscope-us.aliyuncs.com/compatible-mode/v1',
  'https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
  'https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1',
] as const;

const QWEN_3_8_FEATURES = [
  FeatureId.OpenAIUseReasoningEffortParam,
  FeatureId.OpenAIUseThinkingParam3,
  FeatureId.OpenAIUseReasoningContent,
] as const;

function requireModel(id: string) {
  const model = WELL_KNOWN_MODELS.find((candidate) => candidate.id === id);
  if (model === undefined) {
    throw new Error(`${id} model was not found`);
  }
  return model;
}

function requireProvider(name: string) {
  const provider = WELL_KNOWN_PROVIDERS.find(
    (candidate) => candidate.name === name,
  );
  if (provider === undefined) {
    throw new Error(`${name} provider was not found`);
  }
  return provider;
}

function providerConfig(
  baseUrl: string,
  type: ProviderConfig['type'] = 'openai-chat-completion',
): ProviderConfig {
  return {
    type,
    name: 'Qwen feature test',
    baseUrl,
    models: [],
  };
}

describe('GLM-5.3-Flash and Qwen3.8 model catalog', () => {
  it.each([
    {
      id: 'glm-5.3-flash',
      name: 'GLM-5.3-Flash',
      maxInputTokens: 1000000,
      maxOutputTokens: 128000,
      thinking: { type: 'enabled', effort: 'max' },
      parallelToolCalling: undefined,
      capabilities: { toolCalling: true, imageInput: true },
      aliases: ['z-ai/glm-5.3-flash', 'hf:zai-org/GLM-5.3-Flash'],
      efforts: ['max', 'high', 'low'],
      defaultEffort: 'max',
    },
    {
      id: 'qwen3.8-flash-next',
      name: 'Qwen3.8-Flash-Next',
      maxInputTokens: 262144,
      maxOutputTokens: 131072,
      thinking: { type: 'enabled', effort: 'xhigh' },
      parallelToolCalling: true,
      capabilities: { toolCalling: true, imageInput: true },
      aliases: ['hf:Qwen/Qwen3.8-Flash-Next', 'qwen3.8-flash'],
      efforts: ['xhigh', 'medium', 'low'],
      defaultEffort: 'xhigh',
    },
    {
      id: 'qwen3.8-27b',
      name: 'Qwen3.8-27B',
      maxInputTokens: 262144,
      maxOutputTokens: 131072,
      thinking: { type: 'enabled', effort: 'xhigh' },
      parallelToolCalling: undefined,
      capabilities: { toolCalling: true, imageInput: true },
      aliases: ['hf:Qwen/Qwen3.8-27B'],
      efforts: ['xhigh', 'medium', 'low'],
      defaultEffort: 'xhigh',
    },
  ])('declares metadata, aliases, and reasoning presets for $id', (expected) => {
    const model = requireModel(expected.id);
    expect(model).toMatchObject({
      id: expected.id,
      name: expected.name,
      maxInputTokens: expected.maxInputTokens,
      maxOutputTokens: expected.maxOutputTokens,
      stream: true,
      thinking: expected.thinking,
      capabilities: expected.capabilities,
    });
    expect(model.parallelToolCalling).toBe(expected.parallelToolCalling);
    expect(getAlternativeIds(model)).toEqual(expected.aliases);

    const effortTemplate = model.presetTemplates?.find(
      (template) => template.id === 'reasoningEffort',
    );
    expect(effortTemplate?.default).toBe(expected.defaultEffort);
    expect(effortTemplate?.presets.map((preset) => preset.id)).toEqual(
      expected.efforts,
    );
    expect(
      applyPresetTemplateSelections(model, {
        reasoningEffort: expected.efforts.at(-1),
      }).thinking?.effort,
    ).toBe('low');
  });

  it.each([
    'ZhiPu AI',
    'ZhiPu AI (Coding Plan)',
    'Z.AI',
    'Z.AI (Coding Plan)',
  ])('includes GLM-5.3-Flash in %s', (providerName) => {
    const provider = requireProvider(providerName);
    expect(provider.models).toContain('glm-5.3-flash');
    expect(
      isFeatureSupported(
        FeatureId.OpenAIUseGlm53ReasoningEffortParam,
        providerConfig(provider.baseUrl),
        requireModel('glm-5.3-flash'),
      ),
    ).toBe(true);
  });
});

describe('Qwen Cloud provider catalog', () => {
  it.each(QWEN_CLOUD_PROVIDERS)(
    'declares the exact $name preset',
    ({ name, baseUrl, models }) => {
      const provider = requireProvider(name);
      expect({
        name: provider.name,
        type: provider.type,
        baseUrl: provider.baseUrl,
        authTypes: provider.authTypes,
      }).toEqual({
        name,
        type: 'openai-chat-completion',
        baseUrl,
        authTypes: ['api-key'],
      });
      expect(provider.models).toEqual(models);
    },
  );

  it.each(QWEN_CLOUD_PROVIDERS)(
    'resolves every $name model and applies the Qwen cloud override',
    ({ name, models, payAsYouGo }) => {
      const resolved = resolveProviderModels(requireProvider(name));
      expect(resolved.map((model) => model.id).sort()).toEqual(
        [...models].sort(),
      );
      expect(
        resolved.find((model) => model.id === 'qwen3.8-flash'),
      ).toMatchObject({
        id: 'qwen3.8-flash',
        maxInputTokens: 983616,
        maxOutputTokens: 131072,
        thinking: { type: 'enabled', effort: 'xhigh' },
      });

      const qwen27b = resolved.find((model) => model.id === 'qwen3.8-27b');
      if (payAsYouGo) {
        expect(qwen27b).toMatchObject({
          id: 'qwen3.8-27b',
          maxInputTokens: 983616,
          maxOutputTokens: 131072,
        });
      } else {
        expect(qwen27b).toBeUndefined();
      }
    },
  );
});

describe('Qwen3.8 official endpoint feature boundaries', () => {
  const supportedModels: readonly ModelConfig[] = [
    requireModel('qwen3.8-max'),
    requireModel('qwen3.8-flash-next'),
    { ...requireModel('qwen3.8-flash-next'), id: 'qwen3.8-flash' },
    requireModel('qwen3.8-27b'),
  ];

  it.each(OFFICIAL_QWEN_ENDPOINTS)(
    'enables Qwen3.8 protocol features on %s',
    (baseUrl) => {
      for (const featureId of QWEN_3_8_FEATURES) {
        for (const model of supportedModels) {
          expect(
            isFeatureSupported(featureId, providerConfig(baseUrl), model),
            `${featureId} should support ${model.id}`,
          ).toBe(true);
        }
      }
    },
  );

  it.each([
    'qwen3.8-flash-pro',
    'qwen3.8-27b-preview',
    'qwen3.80-flash',
    'vendor/qwen3.8-flash',
    'qwen3.7-plus',
  ])('rejects the near-match model identity %s', (id) => {
    const baseUrl =
      'https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1';
    expect(
      isFeatureSupported(
        FeatureId.OpenAIUseReasoningEffortParam,
        providerConfig(baseUrl),
        { id },
      ),
    ).toBe(false);
  });

  it.each([
    'https://third-party.example/v1',
    'https://dashscope.aliyuncs.com.example/compatible-mode/v1',
    'https://token-plan.us-east-1.maas.aliyuncs.com/compatible-mode/v1',
    'https://token-plan.ap-southeast-1.maas.aliyuncs.com.example/compatible-mode/v1',
  ])('rejects the non-official endpoint %s', (baseUrl) => {
    for (const featureId of QWEN_3_8_FEATURES) {
      expect(
        isFeatureSupported(
          featureId,
          providerConfig(baseUrl),
          requireModel('qwen3.8-flash-next'),
        ),
      ).toBe(false);
    }
  });

  it.each([
    'https://dashscope.aliyuncs.com/apps/anthropic',
    'https://dashscope.aliyuncs.com/compatible-mode/v2',
    'https://token-plan.ap-southeast-1.maas.aliyuncs.com/apps/anthropic',
  ])('does not enable reasoning effort on the wrong path %s', (baseUrl) => {
    expect(
      isFeatureSupported(
        FeatureId.OpenAIUseReasoningEffortParam,
        providerConfig(baseUrl),
        requireModel('qwen3.8-flash-next'),
      ),
    ).toBe(false);
  });

  it('does not enable reasoning effort for a different provider protocol', () => {
    expect(
      isFeatureSupported(
        FeatureId.OpenAIUseReasoningEffortParam,
        providerConfig(OFFICIAL_QWEN_ENDPOINTS[0], 'openai-responses'),
        requireModel('qwen3.8-flash-next'),
      ),
    ).toBe(false);
  });
});
