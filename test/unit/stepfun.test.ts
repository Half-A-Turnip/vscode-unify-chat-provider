import { describe, expect, it, vi } from 'vitest';

vi.mock('vscode', () => {
  class EventEmitter<T> {
    readonly event = (_listener: (value: T) => unknown) => ({
      dispose: () => undefined,
    });
    fire(_value: T): void {}
    dispose(): void {}
  }

  class ThemeIcon {
    constructor(readonly id: string) {}
  }

  return {
    env: { language: 'en' },
    EventEmitter,
    ThemeIcon,
    extensions: { getExtension: () => undefined },
    l10n: {
      t: (message: string | { message: string }) =>
        typeof message === 'string' ? message : message.message,
    },
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
import {
  applyPresetTemplateSelections,
  buildPresetTemplateConfigurationSchema,
} from '../../src/preset-templates';
import type { ProviderConfig } from '../../src/types';
import { mergeWithWellKnownModel } from '../../src/well-known/models';
import {
  resolveProviderModels,
  WELL_KNOWN_PROVIDERS,
} from '../../src/well-known/providers';

describe.each([
  ['StepFun (China)', 'https://api.stepfun.com/v1'],
  ['StepFun (China, Step Plan)', 'https://api.stepfun.com/step_plan/v1'],
  ['StepFun (International)', 'https://api.stepfun.ai/v1'],
  ['StepFun (International, Step Plan)', 'https://api.stepfun.ai/step_plan/v1'],
])('%s Step 5 integration', (name, baseUrl) => {
  const preset = WELL_KNOWN_PROVIDERS.find((provider) => provider.name === name);
  if (!preset) throw new Error(`Missing provider: ${name}`);
  const models = resolveProviderModels(preset);
  const model = models.find((candidate) => candidate.id === 'step-5-preview');
  if (!model) throw new Error(`Missing Step 5 model for ${name}`);
  const config: ProviderConfig = { ...preset, models };

  it('resolves the model on the correct endpoint with StepFun protocol features', () => {
    expect(config.baseUrl).toBe(baseUrl);
    expect(config.type).toBe('openai-chat-completion');
    expect(preset.authTypes).toEqual(['api-key']);
    expect(model.capabilities).toMatchObject({
      toolCalling: true,
      imageInput: true,
    });
    expect(isFeatureSupported(FeatureId.OpenAIOnlyMaxTokens, config, model)).toBe(true);
    expect(isFeatureSupported(FeatureId.OpenAIUseReasoningField, config, model)).toBe(true);
    expect(model.extraBody).toEqual({ reasoning_format: 'general' });
  });

  it('provides the same built-in parameters for manually added or discovered models', () => {
    const merged = mergeWithWellKnownModel({ id: 'step-5-preview' }, config);
    expect(merged.maxInputTokens).toBe(1000000);
    expect(merged.maxOutputTokens).toBe(1000000);
    expect(merged.presetTemplates).toEqual(model.presetTemplates);
    expect(buildPresetTemplateConfigurationSchema(merged)?.properties?.['reasoningEffort'])
      .toMatchObject({ enum: ['high', 'medium', 'low'], default: 'high' });
  });

  it.each(['high', 'medium', 'low'])(
    'translates the %s preset to the native reasoning_effort parameter',
    (effort) => {
      const selected = applyPresetTemplateSelections(model, { reasoningEffort: effort });
      const client = new OpenAIChatCompletionProvider(config);
      const builder: unknown = Reflect.get(client, 'buildReasoningParams');
      if (typeof builder !== 'function') {
        throw new Error('Missing reasoning parameter builder');
      }
      expect(Reflect.apply(builder, client, [selected, 'official']))
        .toEqual({ reasoning_effort: effort });
    },
  );
});
