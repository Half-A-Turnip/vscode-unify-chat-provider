import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AuthTokenInfo } from '../../src/auth/types';
import type { ModelConfig, ProviderConfig } from '../../src/types';

const state = vi.hoisted(() => ({
  configs: [] as unknown[],
  getAvailableModels: vi.fn<(...args: unknown[]) => Promise<ModelConfig[]>>(),
}));

vi.mock('../../src/client/openai/chat-completion-client', () => ({
  OpenAIChatCompletionProvider: class OpenAIChatCompletionProvider {
    constructor(config: unknown) {
      state.configs.push(config);
    }

    async getAvailableModels(...args: unknown[]): Promise<ModelConfig[]> {
      return state.getAvailableModels(...args);
    }
  },
}));

import {
  OpenRouterProvider,
  normalizeOpenRouterCatalogModels,
} from '../../src/client/openrouter/client';

beforeEach(() => {
  state.configs = [];
  state.getAvailableModels.mockReset();
});

describe('OpenRouter model catalog', () => {
  it('normalizes free model names from the models response', () => {
    const paid: ModelConfig = {
      id: 'minimax/minimax-m3',
      name: 'MiniMax: MiniMax M3',
    };

    expect(
      normalizeOpenRouterCatalogModels([
        paid,
        {
          id: 'minimax/minimax-m3:free',
          name: 'MiniMax: MiniMax M3 (free)',
        },
        {
          id: 'example/model:free',
          name: 'Example Model',
        },
      ]),
    ).toEqual([
      paid,
      {
        id: 'minimax/minimax-m3:free',
        name: 'MiniMax: MiniMax M3 (Free)',
      },
      {
        id: 'example/model:free',
        name: 'Example Model (Free)',
      },
    ]);
  });

  it('leaves paid models and free IDs without names unchanged', () => {
    const models: ModelConfig[] = [
      { id: 'example/paid', name: 'Paid (free)' },
      { id: 'example/unknown:free' },
    ];

    const normalized = normalizeOpenRouterCatalogModels(models);

    expect(normalized).toEqual(models);
    expect(normalized[0]).toBe(models[0]);
    expect(normalized[1]).toBe(models[1]);
  });

  it('processes the catalog returned by the base OpenAI client', async () => {
    const catalogModels: ModelConfig[] = [
      {
        id: 'minimax/minimax-m3:free',
        name: 'MiniMax: MiniMax M3 (free)',
      },
    ];
    state.getAvailableModels.mockResolvedValue(catalogModels);
    const config: ProviderConfig = {
      type: 'openrouter',
      name: 'OpenRouter',
      baseUrl: 'https://openrouter.ai/api/v1',
      models: [],
    };
    const credential: AuthTokenInfo = { kind: 'none' };
    const refreshCredential = vi.fn(async () => credential);
    const controller = new AbortController();
    const provider = new OpenRouterProvider(config);

    await expect(
      provider.getAvailableModels(
        credential,
        refreshCredential,
        controller.signal,
      ),
    ).resolves.toEqual([
      {
        id: 'minimax/minimax-m3:free',
        name: 'MiniMax: MiniMax M3 (Free)',
      },
    ]);
    expect(state.configs).toEqual([config]);
    expect(state.getAvailableModels).toHaveBeenCalledWith(
      credential,
      refreshCredential,
      controller.signal,
    );
  });
});
