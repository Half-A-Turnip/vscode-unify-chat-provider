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
import type { ModelConfig, ProviderConfig } from '../../src/types';

const CUSTOM_PROVIDER: ProviderConfig = {
  type: 'openai-chat-completion',
  name: 'Custom OpenAI-compatible endpoint',
  baseUrl: 'https://custom.example/v1',
  models: [],
};

function supportsReasoningContent(model: ModelConfig): boolean {
  return isFeatureSupported(
    FeatureId.OpenAIUseReasoningContent,
    CUSTOM_PROVIDER,
    model,
  );
}

describe('DeepSeek V4 reasoning content feature', () => {
  it.each([
    { id: 'deepseek-v4-pro' },
    { id: 'deepseek-ai/deepseek-v4-pro' },
    { id: 'deepseek-v4-flash#thinking' },
    { id: 'custom-model', family: 'deepseek-v4-pro' },
  ] satisfies ModelConfig[])(
    'enables reasoning_content on custom endpoints for $id',
    (model) => {
      expect(supportsReasoningContent(model)).toBe(true);
    },
  );

  it.each([
    { id: 'deepseek-v3.2' },
    { id: 'ordinary-chat-model' },
    { id: 'custom-model', family: 'deepseek' },
    { id: 'deepseek-v4-pro', family: 'ordinary-chat-model' },
  ] satisfies ModelConfig[])(
    'does not enable reasoning_content for unrelated model $id',
    (model) => {
      expect(supportsReasoningContent(model)).toBe(false);
    },
  );
});
