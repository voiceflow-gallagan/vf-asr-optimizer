import { serve, fetch } from "bun";
import Anthropic from '@anthropic-ai/sdk';

console.log("Starting server...");
console.log("ANTHROPIC_API_KEY:", Bun.env.ANTHROPIC_API_KEY);

interface Conversation {
    user_queries: string[];
    asr_traces: string[];
}

interface ProcessedData {
    conversations: Conversation[];
}

interface OptimizationResponse {
    analysis: string;
    silence_wait: number;
    utterance_end: number;
    punctuation_wait: number;
    no_punctuation_wait: number;
}

function processTranscripts(transcripts: any[], splitByLaunch: boolean = true): ProcessedData {
    let conversations: Conversation[] = [];
    let currentConversation: Conversation = { user_queries: [], asr_traces: [] };

    transcripts.forEach((entry) => {
        if (splitByLaunch && entry.type === "launch") {
            if (currentConversation.user_queries.length > 0 && currentConversation.asr_traces.length > 0) {
                conversations.push({ ...currentConversation });
            }
            currentConversation = { user_queries: [], asr_traces: [] };
        } else if (entry.type === "request") {
            const query = entry.payload?.payload?.query;
            if (query) {
                currentConversation.user_queries.push(query);
            }
        } else if (entry.type === "debug") {
            const message = entry.payload?.payload?.message;
            if (message && message.includes("ASR:")) {
                currentConversation.asr_traces.push(message);
            }
        }
    });

    // Add the last conversation if it has both queries and traces
    if (currentConversation.user_queries.length > 0 && currentConversation.asr_traces.length > 0) {
        conversations.push({ ...currentConversation });
    }

    // If not splitting by launch and we have multiple conversations, combine them
    if (!splitByLaunch && conversations.length > 1) {
        const combinedConversation: Conversation = {
            user_queries: conversations.flatMap(c => c.user_queries),
            asr_traces: conversations.flatMap(c => c.asr_traces)
        };
        return { conversations: [combinedConversation] };
    }

    // Filter out any conversations that might have slipped through without both queries and traces
    conversations = conversations.filter(conv =>
        conv.user_queries.length > 0 && conv.asr_traces.length > 0
    );

    console.log(`Processed ${conversations.length} valid conversations`);
    return { conversations };
}

async function analyzeWithClaude(data: ProcessedData): Promise<OptimizationResponse> {
    const anthropic = new Anthropic({
        apiKey: Bun.env.ANTHROPIC_API_KEY || '',
    });

    const prompt = `Your goal is to analyze previous phone conversation with ASR debug traces to find the optimal settings to set the followings options:

- (ASR) Silence Wait: How much audio silence to wait before resolving, not effective in noisy environments.
- (ASR) Utterance End: Looks for a sufficiently long gap in transcribed word timing.
- (ASR) Punctuation Wait: How long to wait after a full sentence with punctuation is transcribed.
- (ASR) No Punctuation Wait: How long to wait if there is transcription, but no final punctuation.

All values for these settings are in MS and the current config is:

- (ASR) Silence Wait: 500 MS
- (ASR) Utterance End: 1500 MS
- (ASR) Punctuation Wait: 1000 MS
- (ASR) No Punctuation Wait: 5000 MS

Now, based on the following logs, please provide the optimal settings with a summary to justify your choices:

${JSON.stringify(data, null, 2)}

IMPORTANT: Your response must be a valid JSON object with exactly these fields:
{
    "analysis": "your analysis here",
    "silence_wait": number,
    "utterance_end": number,
    "punctuation_wait": number,
    "no_punctuation_wait": number
}`;

    const message = await anthropic.messages.create({
        model: 'claude-3-sonnet-20240229',
        max_tokens: 1024,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.5,
        system: "You are a JSON-only response bot. You must respond with valid JSON that matches the schema specified in the prompt."
    });

    try {
        const content = message.content[0].type === 'text'
            ? message.content[0].text
            : JSON.stringify(message.content[0]);
        console.log("Claude response:", content);

        // Extract JSON object from the response
        const jsonStartIndex = content.indexOf('{');
        const jsonEndIndex = content.lastIndexOf('}') + 1;
        const jsonString = content.substring(jsonStartIndex, jsonEndIndex);

        const parsed = JSON.parse(jsonString.trim());

        if (!parsed.analysis ||
            typeof parsed.silence_wait !== 'number' ||
            typeof parsed.utterance_end !== 'number' ||
            typeof parsed.punctuation_wait !== 'number' ||
            typeof parsed.no_punctuation_wait !== 'number') {
            throw new Error('Response missing required fields');
        }

        return parsed as OptimizationResponse;
    } catch (error) {
        console.error("Parse error:", error);
        const content = message.content[0].type === 'text'
            ? message.content[0].text
            : JSON.stringify(message.content[0]);
        console.error("Raw response:", content);
        throw new Error(`Failed to parse Claude response as JSON: ${error.message}`);
    }
}

const server = serve({
    port: 3000,
    async fetch(req) {
        const url = new URL(req.url);

        if (url.pathname === "/optimize") {
            console.log("Received request to /optimize");

            const authHeader = req.headers.get("Authorization");
            console.log("Auth header:", authHeader);

            if (!authHeader) {
                return new Response("Authorization header is required", { status: 401 });
            }

            if (!process.env.ANTHROPIC_API_KEY) {
                return new Response("ANTHROPIC_API_KEY environment variable is required", { status: 500 });
            }

            const projectID = url.searchParams.get("projectID");
            const encodedUserID = url.searchParams.get("userID");

            if (!projectID || !encodedUserID) {
                return new Response("projectID and userID are required query parameters", { status: 400 });
            }

            // Clean and decode the userID to handle '+' in phone numbers
            const userID = encodedUserID
                .replace(/\s/g, '+')  // Replace spaces with +
                .replace(/\+$/, '')   // Remove trailing + if present (from curl)
                .replace(/^([^+])/, '+$1'); // Add + prefix if missing

            console.log("Original userID param:", encodedUserID);
            console.log("Processed userID:", userID);

            try {
                // First, fetch the list of transcripts to find the matching sessionID
                const transcriptsListUrl = `https://api.voiceflow.com/v2/transcripts/${projectID}?range=Last%207%20Days`;
                console.log("Fetching transcripts list from:", transcriptsListUrl);

                const listResponse = await fetch(transcriptsListUrl, {
                    headers: {
                        Authorization: authHeader,
                        'Content-Type': 'application/json'
                    }
                });

                if (!listResponse.ok) {
                    const errorText = await listResponse.text();
                    console.error("Voiceflow API error:", errorText);
                    return new Response(`Failed to fetch transcripts list: ${errorText}`, {
                        status: listResponse.status,
                        headers: { 'Content-Type': 'application/json' }
                    });
                }

                const transcriptsList = await listResponse.json();
                const matchingTranscript = transcriptsList.find((t: any) => t.sessionID === userID);

                if (!matchingTranscript) {
                    return new Response(`No transcript found for userID: ${userID}`, {
                        status: 404,
                        headers: { 'Content-Type': 'application/json' }
                    });
                }

                const transcriptID = matchingTranscript._id;
                const transcriptsUrl = `https://api.voiceflow.com/v2/transcripts/${projectID}/${transcriptID}`;
                console.log("Fetching transcript details from:", transcriptsUrl);

                const response = await fetch(transcriptsUrl, {
                    headers: {
                        Authorization: authHeader,
                        'Content-Type': 'application/json'
                    }
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error("Voiceflow API error:", errorText);
                    return new Response(`Failed to fetch transcripts: ${errorText}`, {
                        status: response.status,
                        headers: { 'Content-Type': 'application/json' }
                    });
                }

                const transcripts = await response.json();
                const splitByLaunch = url.searchParams.get("splitByLaunch") !== "false";
                const processedData = processTranscripts(transcripts, splitByLaunch);
                const optimization = await analyzeWithClaude(processedData);

                return new Response(JSON.stringify(optimization), {
                    headers: { "Content-Type": "application/json" }
                });
            } catch (error) {
                console.error("Server error:", error);
                return new Response(JSON.stringify({ error: error.message }), {
                    status: 500,
                    headers: { 'Content-Type': 'application/json' }
                });
            }
        }

        return new Response("Not Found", { status: 404 });
    },
});

console.log(`Server running at http://localhost:${server.port}`);
