import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI, { AzureOpenAI } from "openai"
import { withRetry } from "../retry"
import { ApiHandlerOptions, azureOpenAiDefaultApiVersion, ModelInfo, openAiModelInfoSaneDefaults } from "../../shared/api"
import { ApiHandler } from "../index"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream } from "../transform/stream"
import { convertToR1Format } from "../transform/r1-format"
import { ChatCompletionReasoningEffort } from "openai/resources/chat/completions.mjs"
import { ApiStreamChunk } from "../transform/stream"

export class OpenAiHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: OpenAI

	constructor(options: ApiHandlerOptions) {
		if (options.openAiConfigs && options.openAiConfigs.length > 0) {
			const index = options.openAiSelectedConfigIndex ?? 0
			const config = options.openAiConfigs[index]
			options.openAiBaseUrl = config.openAiBaseUrl
			options.openAiApiKey = config.openAiApiKey
			options.openAiModelId = config.openAiModelId
			options.openAiModelInfo = config.openAiModelInfo
			options.azureApiVersion = config.azureApiVersion
		}
		this.options = options
		// Azure API shape slightly differs from the core API shape: https://github.com/openai/openai-node?tab=readme-ov-file#microsoft-azure-openai
		// Use azureApiVersion to determine if this is an Azure endpoint, since the URL may not always contain 'azure.com'
		if (
			this.options.azureApiVersion ||
			(this.options.openAiBaseUrl?.toLowerCase().includes("azure.com") &&
				!this.options.openAiModelId?.toLowerCase().includes("deepseek"))
		) {
			this.client = new AzureOpenAI({
				baseURL: this.options.openAiBaseUrl,
				apiKey: this.options.openAiApiKey,
				apiVersion: this.options.azureApiVersion || azureOpenAiDefaultApiVersion,
			})
		} else {
			this.client = new OpenAI({
				baseURL: this.options.openAiBaseUrl,
				apiKey: this.options.openAiApiKey,
			})
		}
	}

	private processAzureDeepseekContent(
		content: string,
		inReasoning: boolean,
	): { events: ApiStreamChunk[]; inReasoning: boolean } {
		const openTag = "<think>"
		const closeTag = "</think>"
		const events: ApiStreamChunk[] = []

		if (content.includes(openTag)) {
			const parts = content.split(openTag)
			inReasoning = true
			if (parts.length > 1 && parts[1].length > 0) {
				events.push({ type: "reasoning", reasoning: parts[1] })
			}
		} else if (content.includes(closeTag)) {
			const parts = content.split(closeTag)
			if (parts.length > 0 && parts[0].length > 0) {
				events.push({ type: "reasoning", reasoning: parts[0] })
			}
			inReasoning = false
			if (parts.length > 1 && parts[1].length > 0) {
				events.push({ type: "text", text: parts[1] })
			}
		} else {
			if (inReasoning) {
				events.push({ type: "reasoning", reasoning: content })
			} else {
				events.push({ type: "text", text: content })
			}
		}

		return { events, inReasoning }
	}

	@withRetry()
	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const modelId = this.options.openAiModelId ?? ""
		const isDeepseekReasoner = modelId.includes("deepseek-reasoner") || modelId.includes("deepseek-r1")
		const isAzureDeepseek =
			(this.options.azureApiVersion || this.options.openAiBaseUrl?.toLowerCase().includes("azure.com")) &&
			isDeepseekReasoner
		const isO3Mini = modelId.includes("o3-mini")

		let openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]
		let temperature: number | undefined = this.options.openAiModelInfo?.temperature ?? openAiModelInfoSaneDefaults.temperature
		let reasoningEffort: ChatCompletionReasoningEffort | undefined = undefined

		if (isDeepseekReasoner) {
			openAiMessages = convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])
		}

		if (isO3Mini) {
			openAiMessages = [{ role: "developer", content: systemPrompt }, ...convertToOpenAiMessages(messages)]
			temperature = undefined // does not support temperature
			reasoningEffort = (this.options.o3MiniReasoningEffort as ChatCompletionReasoningEffort) || "medium"
		}

		const stream = await this.client.chat.completions.create({
			model: modelId,
			messages: openAiMessages,
			temperature,
			reasoning_effort: reasoningEffort,
			stream: true,
			stream_options: { include_usage: true },
		})

		let inReasoning = false
		for await (const chunk of stream) {
			const delta = chunk.choices[0]?.delta
			if (delta?.content) {
				if (isAzureDeepseek) {
					const result = this.processAzureDeepseekContent(delta.content, inReasoning)
					inReasoning = result.inReasoning
					for (const event of result.events) {
						yield event
					}
				} else {
					yield { type: "text", text: delta.content }
				}
			}

			if (delta && "reasoning_content" in delta && delta.reasoning_content) {
				yield {
					type: "reasoning",
					reasoning: (delta.reasoning_content as string | undefined) || "",
				}
			}

			if (chunk.usage) {
				yield {
					type: "usage",
					inputTokens: chunk.usage.prompt_tokens || 0,
					outputTokens: chunk.usage.completion_tokens || 0,
				}
			}
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.openAiModelId ?? "",
			info: this.options.openAiModelInfo ?? openAiModelInfoSaneDefaults,
		}
	}
}
