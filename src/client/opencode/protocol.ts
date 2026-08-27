import { getBaseModelId } from '../../model-id-utils';
import type { ModelConfig } from '../../types';

export type OpenCodeService = 'zen' | 'go';

export type OpenCodeProtocol =
  | 'anthropic-messages'
  | 'google-generative-ai'
  | 'openai-chat-completions'
  | 'openai-responses';

const RESPONSES_MODEL_PREFIXES = ['gpt-', 'grok-', 'muse-'] as const;

function normalizedModelId(model: Pick<ModelConfig, 'id'>): string {
  return getBaseModelId(model.id).trim().toLowerCase();
}

function startsWithAny(value: string, prefixes: readonly string[]): boolean {
  return prefixes.some((prefix) => value.startsWith(prefix));
}

/**
 * Resolve the native OpenCode endpoint documented for a model family.
 * Unknown IDs inherit OpenCode's OpenAI-compatible default.
 */
export function resolveOpenCodeProtocol(
  service: OpenCodeService,
  model: Pick<ModelConfig, 'id'>,
): OpenCodeProtocol {
  const modelId = normalizedModelId(model);

  if (service === 'zen' && modelId.startsWith('gemini-')) {
    return 'google-generative-ai';
  }

  if (startsWithAny(modelId, RESPONSES_MODEL_PREFIXES)) {
    return 'openai-responses';
  }

  if (modelId.startsWith('claude-') || modelId.startsWith('qwen')) {
    return 'anthropic-messages';
  }

  if (service === 'go' && modelId.startsWith('minimax-')) {
    return 'anthropic-messages';
  }

  // Historical Zen free MiniMax variants use the Messages endpoint while
  // paid MiniMax models use Chat Completions.
  if (
    service === 'zen' &&
    modelId.startsWith('minimax-') &&
    modelId.endsWith('-free')
  ) {
    return 'anthropic-messages';
  }

  return 'openai-chat-completions';
}
