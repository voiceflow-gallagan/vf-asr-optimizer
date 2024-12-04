import { serve, fetch } from "bun";
import Anthropic from '@anthropic-ai/sdk';
import { Database } from 'bun:sqlite';
import { mkdir } from "node:fs/promises";

console.log("Starting server...");


// Initialize database
const DB_PATH = Bun.env.DB_PATH || './data/optimization_results.db';
const DB_DIR = DB_PATH.substring(0, DB_PATH.lastIndexOf('/'));

let db: Database | undefined;

try {
    // Check if database path exists
    console.log('Checking database path:', DB_PATH);
    if (!await Bun.file(DB_DIR).exists()) {
        console.log('Creating database directory:', DB_DIR);
        await mkdir(DB_DIR, { recursive: true });
        // Ensure directory has proper permissions
        await Bun.write(DB_DIR + '/.keep', '');
        console.log('Database directory created successfully');
    }

    console.log('Initializing database connection...');
    db = new Database(DB_PATH, { create: true });

    // Configure database settings
    console.log('Configuring database settings...');
    db.run('PRAGMA journal_mode = WAL');
    db.run('PRAGMA synchronous = NORMAL');
    console.log('Database write mode configured');

    console.log('Creating table if not exists...');
    db.run(`CREATE TABLE IF NOT EXISTS optimization_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        result TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'pending'
    )`);
    console.log('Database initialized successfully');
} catch (dbError) {
    console.error('Failed to initialize database:', dbError);
    if (db) {
        db.close();
    }
    process.exit(1);
}

// Add process cleanup
process.on('exit', () => {
    if (db) {
        console.log('Closing database connection on exit');
        db.close();
    }
});

process.on('SIGINT', () => {
    if (db) {
        console.log('Closing database connection on SIGINT');
        db.close();
    }
    process.exit(0);
});


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

interface OptimizeRequest {
    projectID: string;
    userID: string;
    vfApiKey: string;
    splitByLaunch?: boolean;
}

interface OptimizationResult {
    id: number;
    user_id: string;
    project_id: string;
    result: string;  // Store as string in DB
    created_at: string;
    status: 'pending' | 'completed' | 'error';
}

interface ParsedOptimizationResult extends Omit<OptimizationResult, 'result'> {
    result: OptimizationResponse | { error: string } | { status: string };
}

function formatUserID(userID: string): string {
    return userID
        .replace(/\s/g, '+')  // Replace spaces with +
        .replace(/\+$/, '')   // Remove trailing +
        .replace(/^([^+])/, '+$1'); // Add + prefix if missing
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

    console.log("Analyzing with Claude...");
    console.log(data);

    const prompt = `Your goal is to analyze previous phone conversation with ASR debug traces to find the optimal settings to set the followings options:

- (ASR) Silence Wait: How much audio silence to wait before resolving, not effective in noisy environments.
- (ASR) Utterance End: Looks for a sufficiently long gap in transcribed word timing.
- (ASR) Punctuation Wait: How long to wait after a full sentence with punctuation is transcribed.
- (ASR) No Punctuation Wait: How long to wait if there is transcription, but no final punctuation.

All values for these settings are in MS and the default config is:

- (ASR) Silence Wait: 500 MS
- (ASR) Utterance End: 1500 MS
- (ASR) Punctuation Wait: 1000 MS
- (ASR) No Punctuation Wait: 5000 MS

Now, based on the following logs, please provide the optimal settings with a summary to justify your choices:

<logs>
${JSON.stringify(data, null, 2)}
</logs>


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
        max_tokens: 2024,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.1,
        system: "You are a JSON-only response bot. You must respond with valid JSON that matches the schema specified in the prompt."
    });

    const content = message.content[0].type === 'text'
        ? message.content[0].text
        : JSON.stringify(message.content[0]);

    const jsonStartIndex = content.indexOf('{');
    const jsonEndIndex = content.lastIndexOf('}') + 1;
    const jsonString = content.substring(jsonStartIndex, jsonEndIndex);

    return JSON.parse(jsonString.trim());
}

async function processOptimization(userID: string, projectID: string, vfApiKey: string, splitByLaunch: boolean) {
    if (!db) {
        throw new Error('Database not initialized');
    }

    try {
        // First, fetch the list of transcripts to find the matching sessionID
        const transcriptsListUrl = `https://api.voiceflow.com/v2/transcripts/${projectID}?range=Last%207%20Days`;
        console.log("Fetching transcripts list from:", transcriptsListUrl);

        const listResponse = await fetch(transcriptsListUrl, {
            headers: {
                Authorization: vfApiKey,
                'Content-Type': 'application/json'
            }
        });

        if (!listResponse.ok) {
            throw new Error(`Failed to fetch transcripts list: ${await listResponse.text()}`);
        }

        const transcriptsList = await listResponse.json();
        const matchingTranscript = transcriptsList.find((t: any) => t.sessionID === userID);

        if (!matchingTranscript) {
            throw new Error(`No transcript found for userID: ${userID}`);
        }

        const transcriptID = matchingTranscript._id;
        const transcriptsUrl = `https://api.voiceflow.com/v2/transcripts/${projectID}/${transcriptID}`;

        const response = await fetch(transcriptsUrl, {
            headers: {
                Authorization: vfApiKey,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch transcripts: ${await response.text()}`);
        }

        const transcripts = await response.json();
        const processedData = processTranscripts(transcripts, splitByLaunch);
        const optimization = await analyzeWithClaude(processedData);

        console.log('Starting database update...');

        // Update the result directly without transaction
        const updateResult = db.run(
            `UPDATE optimization_results
             SET result = ?, status = ?
             WHERE user_id = ? AND project_id = ?`,
            [JSON.stringify(optimization), 'completed', userID, projectID]
        );

        console.log('Database update result:', updateResult);

        // Verify the update
        const verifyResult = db.query<OptimizationResult>(
            'SELECT * FROM optimization_results WHERE user_id = ? AND project_id = ?'
        ).get(userID, projectID);

        console.log('Verification query result:', verifyResult);

        if (!verifyResult || verifyResult.status !== 'completed') {
            throw new Error('Failed to verify database update');
        }

        console.log('Database update completed successfully');
        return optimization;
    } catch (error) {
        console.error('Error in processOptimization:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';

        if (db) {
            try {
                console.log('Updating error status...');
                const errorResult = db.run(
                    `UPDATE optimization_results
                     SET result = ?, status = ?
                     WHERE user_id = ? AND project_id = ?`,
                    [JSON.stringify({ error: errorMessage }), 'error', userID, projectID]
                );
                console.log('Error status update result:', errorResult);
            } catch (dbError) {
                console.error('Failed to update error status:', dbError);
            }
        }
        throw error;
    }
}

const server = serve({
    port: Bun.env.PORT || 3000,
    async fetch(req) {
        const url = new URL(req.url);

        // New endpoint to fetch results
        if (url.pathname === "/results" && req.method === "GET") {
            const rawUserID = url.searchParams.get("userID");
            if (!rawUserID) {
                return new Response("userID is required as a query parameter", {
                    status: 400,
                    headers: { 'Content-Type': 'application/json' }
                });
            }

            const userID = formatUserID(rawUserID);
            console.log("Fetching results for userID:", userID);

            // Get the record
            const result = db.query<OptimizationResult, string>(
                "SELECT * FROM optimization_results WHERE user_id = ?"
            ).get(userID);

            if (!result) {
                return new Response(JSON.stringify({ error: "No results found" }), {
                    status: 404,
                    headers: { 'Content-Type': 'application/json' }
                });
            }

            // Parse the JSON string in result
            const parsedResult: ParsedOptimizationResult = {
                ...result,
                result: JSON.parse(result.result)
            };

            console.log("Found result:", parsedResult);
            return new Response(JSON.stringify(parsedResult), {
                headers: { 'Content-Type': 'application/json' }
            });
        }

        if (url.pathname === "/optimize" && req.method === "POST") {
            console.log("Received POST request to /optimize");

            try {
                const body = await req.json() as OptimizeRequest;
                const { projectID, userID: encodedUserID, vfApiKey, splitByLaunch = true } = body;

                if (!projectID || !encodedUserID || !vfApiKey) {
                    return new Response("projectID, userID, and vfApiKey are required in request body", {
                        status: 400,
                        headers: { 'Content-Type': 'application/json' }
                    });
                }

                if (!process.env.ANTHROPIC_API_KEY) {
                    return new Response("ANTHROPIC_API_KEY environment variable is required", { status: 500 });
                }

                // Clean and decode the userID
                const userID = formatUserID(encodedUserID);
                console.log("Processing request for userID:", userID);

                // Update or insert initial pending record
                try {
                    db.run(
                        `INSERT OR REPLACE INTO optimization_results (user_id, project_id, result, status)
                         VALUES (?, ?, ?, ?)`,
                        [userID, projectID, JSON.stringify({ status: "processing" }), "pending"]
                    );
                } catch (dbError) {
                    console.error("Database error:", dbError);
                    return new Response(JSON.stringify({ error: "Failed to initialize optimization record" }), {
                        status: 500,
                        headers: { 'Content-Type': 'application/json' }
                    });
                }

                // Run the optimization process in the background
                (async () => {
                    try {
                        console.log(`Starting background optimization for userID: ${userID}`);
                        const result = await processOptimization(userID, projectID, vfApiKey, splitByLaunch);
                        console.log(`Optimization completed successfully for userID: ${userID}`, result);
                    } catch (error) {
                        console.error(`Optimization error for userID: ${userID}:`, error);
                        try {
                            console.log(`Updating database with error status for userID: ${userID}`);
                            db.run(
                                `INSERT OR REPLACE INTO optimization_results (user_id, project_id, result, status)
                                 VALUES (?, ?, ?, ?)`,
                                [userID, projectID, JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error occurred' }), "error"]
                            );
                            console.log(`Successfully updated error status for userID: ${userID}`);
                        } catch (dbError) {
                            console.error(`Failed to update error status for userID: ${userID}:`, dbError);
                        }
                    }
                })();

                return new Response(JSON.stringify({
                    message: "Optimization started",
                    userID: userID
                }), {
                    status: 202,
                    headers: { 'Content-Type': 'application/json' }
                });
            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
                console.error("Request error:", errorMessage);
                return new Response(JSON.stringify({ error: errorMessage }), {
                    status: 400,
                    headers: { 'Content-Type': 'application/json' }
                });
            }
        }

        return new Response("Not Found", { status: 404 });
    },
});

console.log(`Server running at http://localhost:${server.port}`);
