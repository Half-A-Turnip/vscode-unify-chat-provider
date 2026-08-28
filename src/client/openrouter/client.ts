import type { AuthTokenInfo, AuthTokenRefresh } from '../../auth/types';
import type { ModelConfig } from '../../types';
import { OpenAIChatCompletionProvider } from '../openai/chat-completion-client';

const OPENROUTER_FREE_MODEL_SUFFIX = ':free';
const FREE_NAME_SUFFIX = /\s*\(free\)\s*$/i;

function normalizeFreeName(name: string): string {
  const trimmed = name.trim();
  const baseName = trimmed.replace(FREE_NAME_SUFFIX, '').trim();

  if (baseName !== trimmed) {
    return baseName ? `${baseName} (Free)` : '(Free)';
  }
  if (/\bfree\b/i.test(trimmed)) {
    return trimmed;
  }
  return `${trimmed} (Free)`;
}

export function normalizeOpenRouterCatalogModels(
  models: ModelConfig[],
): ModelConfig[] {
  return models.map((model) => {
    const name = model.name?.trim();
    if (
      !model.id.toLowerCase().endsWith(OPENROUTER_FREE_MODEL_SUFFIX) ||
      !name
    ) {
      return model;
    }

    const normalizedName = normalizeFreeName(name);
    return normalizedName === model.name
      ? model
      : { ...model, name: normalizedName };
  });
}

export class OpenRouterProvider extends OpenAIChatCompletionProvider {
  override async getAvailableModels(
    credential: AuthTokenInfo,
    refreshCredential?: AuthTokenRefresh,
    signal?: AbortSignal,
  ): Promise<ModelConfig[]> {
    const models = await super.getAvailableModels(
      credential,
      refreshCredential,
      signal,
    );
    return normalizeOpenRouterCatalogModels(models);
  }
}
