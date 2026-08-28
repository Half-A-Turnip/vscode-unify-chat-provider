import type {
  CancellationToken,
  LanguageModelChatRequestMessage,
  LanguageModelResponsePart2,
  ProvideLanguageModelChatResponseOptions,
} from 'vscode';
import type { AuthTokenInfo, AuthTokenRefresh } from '../../auth/types';
import type { RequestLogger } from '../../logger';
import type { ChatRequestTrace, ModelConfig, ProviderConfig } from '../../types';
import { AnthropicProvider } from '../anthropic/client';
import { GoogleAIStudioProvider } from '../google/ai-studio-client';
import type {
  ApiProvider,
  EmptyChatResponsePolicy,
} from '../interface';
import { OpenAIChatCompletionProvider } from '../openai/chat-completion-client';
import { OpenAIResponsesProvider } from '../openai/responses-client';
import {
  resolveOpenCodeProtocol,
  type OpenCodeProtocol,
  type OpenCodeService,
} from './protocol';

export const OPENCODE_ZEN_API_BASE_URL = 'https://opencode.ai/zen/v1';
export const OPENCODE_GO_API_BASE_URL = 'https://opencode.ai/zen/go/v1';

export interface OpenCodeDelegateConfigs {
  readonly anthropic: ProviderConfig;
  readonly google: ProviderConfig;
  readonly openAIChat: ProviderConfig;
  readonly openAIResponses: ProviderConfig;
}

async function distinguishFreeCatalogModels(
  models: ModelConfig[],
): Promise<ModelConfig[]> {
  if (!models.some((model) => model.id.toLowerCase().endsWith('-free'))) {
    return models;
  }

  // Load model metadata only when the OpenCode catalog contains free variants.
  // This keeps provider-specific labeling out of the shared merge path and
  // avoids a definitions -> provider -> well-known-models module cycle.
  const { findBestMatchingWellKnownModel } = await import(
    '../../well-known/models'
  );

  return models.map((model) => {
    if (!model.id.toLowerCase().endsWith('-free')) {
      return model;
    }

    const displayName =
      model.name?.trim() ||
      findBestMatchingWellKnownModel(model.id)?.name?.trim();
    if (!displayName || /\bfree\b/i.test(displayName)) {
      return model;
    }

    return { ...model, name: `${displayName} (Free)` };
  });
}

function getOpenCodeApiBaseUrl(service: OpenCodeService): string {
  return service === 'zen'
    ? OPENCODE_ZEN_API_BASE_URL
    : OPENCODE_GO_API_BASE_URL;
}

export function createOpenCodeDelegateConfigs(
  config: ProviderConfig,
  service: OpenCodeService,
): OpenCodeDelegateConfigs {
  const shared: ProviderConfig = {
    ...config,
    baseUrl: getOpenCodeApiBaseUrl(service),
    useRawBaseUrl: false,
  };

  return {
    anthropic: { ...shared, type: 'anthropic' },
    google: { ...shared, type: 'google-ai-studio' },
    openAIChat: { ...shared, type: 'openai-chat-completion' },
    openAIResponses: { ...shared, type: 'openai-responses' },
  };
}

class OpenCodeProvider implements ApiProvider {
  private readonly anthropicProvider: AnthropicProvider;
  private readonly googleProvider: GoogleAIStudioProvider;
  private readonly openAIChatProvider: OpenAIChatCompletionProvider;
  private readonly openAIResponsesProvider: OpenAIResponsesProvider;

  constructor(
    config: ProviderConfig,
    private readonly service: OpenCodeService,
  ) {
    const delegateConfigs = createOpenCodeDelegateConfigs(config, service);
    this.anthropicProvider = new AnthropicProvider(delegateConfigs.anthropic);
    this.googleProvider = new GoogleAIStudioProvider(delegateConfigs.google);
    this.openAIChatProvider = new OpenAIChatCompletionProvider(
      delegateConfigs.openAIChat,
    );
    this.openAIResponsesProvider = new OpenAIResponsesProvider(
      delegateConfigs.openAIResponses,
    );
  }

  getEmptyChatResponsePolicy(model: ModelConfig): EmptyChatResponsePolicy {
    return resolveOpenCodeProtocol(this.service, model) ===
      'anthropic-messages'
      ? 'success'
      : 'retry';
  }

  private getDelegate(protocol: OpenCodeProtocol): ApiProvider {
    switch (protocol) {
      case 'anthropic-messages':
        return this.anthropicProvider;
      case 'google-generative-ai':
        return this.googleProvider;
      case 'openai-responses':
        return this.openAIResponsesProvider;
      case 'openai-chat-completions':
        return this.openAIChatProvider;
    }
  }

  async *streamChat(
    encodedModelId: string,
    model: ModelConfig,
    messages: readonly LanguageModelChatRequestMessage[],
    options: ProvideLanguageModelChatResponseOptions,
    requestTrace: ChatRequestTrace,
    token: CancellationToken,
    logger: RequestLogger,
    credential: AuthTokenInfo,
    refreshCredential?: AuthTokenRefresh,
  ): AsyncGenerator<LanguageModelResponsePart2> {
    const delegate = this.getDelegate(
      resolveOpenCodeProtocol(this.service, model),
    );
    yield* delegate.streamChat(
      encodedModelId,
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

  estimateTokenCount(text: string): number {
    return this.openAIChatProvider.estimateTokenCount(text);
  }

  async getAvailableModels(
    credential: AuthTokenInfo,
    refreshCredential?: AuthTokenRefresh,
    signal?: AbortSignal,
  ): Promise<ModelConfig[]> {
    const models = await this.openAIChatProvider.getAvailableModels(
      credential,
      refreshCredential,
      signal,
    );
    return distinguishFreeCatalogModels(models);
  }
}

export class OpenCodeZenProvider extends OpenCodeProvider {
  constructor(config: ProviderConfig) {
    super(config, 'zen');
  }
}

export class OpenCodeGoProvider extends OpenCodeProvider {
  constructor(config: ProviderConfig) {
    super(config, 'go');
  }
}

export { resolveOpenCodeProtocol, type OpenCodeProtocol, type OpenCodeService };
