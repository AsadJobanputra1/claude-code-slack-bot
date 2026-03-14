import { beforeEach, describe, expect, it, vi } from 'vitest';
import { query } from '@anthropic-ai/claude-agent-sdk';
import { ClaudeHandler } from './claude-handler';

vi.mock('@anthropic-ai/claude-agent-sdk', () => ({
  query: vi.fn(),
}));

type AnyMessage = any;

function createMcpManagerStub() {
  return {
    getServerConfiguration: () => ({
      filesystem: {
        command: 'npx',
        args: ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'],
      },
    }),
    getDefaultAllowedTools: () => ['mcp__filesystem'],
  };
}

async function* streamMessages(messages: AnyMessage[]): AsyncGenerator<AnyMessage, void, unknown> {
  for (const message of messages) {
    yield message;
  }
}

function makeInitMessage(sessionId: string): AnyMessage {
  return {
    type: 'system',
    subtype: 'init',
    session_id: sessionId,
    model: 'claude-sonnet-4-6',
    tools: ['Read'],
    mcp_servers: [],
    apiKeySource: 'user',
    cwd: '/tmp',
    permissionMode: 'default',
    claude_code_version: '2.1.74',
    slash_commands: [],
    output_style: 'default',
    skills: [],
    plugins: [],
  };
}

describe('ClaudeHandler.streamQuery', () => {
  const mockedQuery = vi.mocked(query);

  beforeEach(() => {
    mockedQuery.mockReset();
  });

  it('passes expected Agent SDK options including settingSources and MCP merge', async () => {
    mockedQuery.mockImplementation(() => streamMessages([makeInitMessage('session-1')]) as any);
    const handler = new ClaudeHandler(createMcpManagerStub() as any);

    const session = handler.createSession('U1', 'C1', 'T1');
    const output: AnyMessage[] = [];

    for await (const message of handler.streamQuery(
      'hello',
      session,
      undefined,
      '/repo/project',
      { channel: 'C1', threadTs: 'T1', user: 'U1' }
    )) {
      output.push(message);
    }

    expect(output).toHaveLength(1);
    expect(mockedQuery).toHaveBeenCalledTimes(1);

    const params = mockedQuery.mock.calls[0][0] as any;
    expect(params.prompt).toBe('hello');
    expect(params.options.cwd).toBe('/repo/project');
    expect(params.options.settingSources).toEqual(['user', 'project', 'local']);
    expect(params.options.permissionMode).toBe('default');
    expect(params.options.permissionPromptToolName).toBe('mcp__permission-prompt__permission_prompt');
    expect(params.options.mcpServers.filesystem).toBeDefined();
    expect(params.options.mcpServers['permission-prompt']).toBeDefined();
    expect(params.options.allowedTools).toContain('mcp__filesystem');
    expect(params.options.allowedTools).toContain('mcp__permission-prompt');
  });

  it('forwards resume when session has sessionId', async () => {
    mockedQuery.mockImplementation(() => streamMessages([]) as any);
    const handler = new ClaudeHandler(createMcpManagerStub() as any);

    const session = handler.createSession('U1', 'C1', 'T1');
    session.sessionId = 'resume-session-123';

    for await (const _ of handler.streamQuery('resume me', session)) {
      // no-op
    }

    const params = mockedQuery.mock.calls[0][0] as any;
    expect(params.options.resume).toBe('resume-session-123');
  });

  it('stores session id from init message', async () => {
    mockedQuery.mockImplementation(() => streamMessages([makeInitMessage('new-session-id')]) as any);
    const handler = new ClaudeHandler(createMcpManagerStub() as any);
    const session = handler.createSession('U1', 'C1', 'T1');

    for await (const _ of handler.streamQuery('start', session)) {
      // no-op
    }

    expect(session.sessionId).toBe('new-session-id');
  });

  it('yields assistant and result messages from SDK stream', async () => {
    const init = makeInitMessage('stream-session');
    const assistant = {
      type: 'assistant',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'hello from claude' }],
      },
      parent_tool_use_id: null,
      session_id: 'stream-session',
    };
    const result = {
      type: 'result',
      subtype: 'success',
      duration_ms: 1,
      duration_api_ms: 1,
      is_error: false,
      num_turns: 1,
      result: 'done',
      session_id: 'stream-session',
      total_cost_usd: 0,
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_creation_input_tokens: null,
        cache_read_input_tokens: null,
        server_tool_use: null,
        service_tier: null,
        cache_creation: null,
      },
    };

    mockedQuery.mockImplementation(() => streamMessages([init, assistant, result]) as any);
    const handler = new ClaudeHandler(createMcpManagerStub() as any);
    const session = handler.createSession('U1', 'C1', 'T1');

    const output: AnyMessage[] = [];
    for await (const message of handler.streamQuery('stream', session)) {
      output.push(message);
    }

    expect(output).toEqual([init, assistant, result]);
  });

  it('rethrows errors from SDK query', async () => {
    mockedQuery.mockImplementation(() => {
      throw new Error('boom');
    });
    const handler = new ClaudeHandler(createMcpManagerStub() as any);
    const session = handler.createSession('U1', 'C1', 'T1');

    await expect(
      (async () => {
        for await (const _ of handler.streamQuery('fails', session)) {
          // no-op
        }
      })()
    ).rejects.toThrow('boom');
  });
});
