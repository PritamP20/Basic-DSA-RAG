import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import * as dotenv from "dotenv";
dotenv.config();
import readlineSync from "readline-sync";
import { Pinecone } from "@pinecone-database/pinecone";

const ai = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-2.0-flash-exp",
  temperature: 0.1,
});

const History = [];

async function transformQuery(question) {
  try {
    // Create context from history for query transformation
    const historyContext =
      History.length > 0
        ? History.map((h) => `${h.role}: ${h.content}`).join("\n")
        : "No previous conversation history.";

    const fullPrompt = `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
Only output the rewritten question and nothing else.

Chat History:
${historyContext}

Follow Up user Question: ${question}`;

    const response = await ai.invoke([{ role: "user", content: fullPrompt }]);

    // Return the transformed query as a string
    return response.content || question; // Fallback to original question if no response
  } catch (error) {
    console.error("Error in transformQuery:", error);
    // Return original question if transformation fails
    return question;
  }
}

async function chatting(question) {
  try {
    console.log("Transforming query...");

    // Transform the query first
    const transformedQuery = await transformQuery(question);
    console.log("Transformed query:", transformedQuery);

    console.log("Creating embeddings...");

    // Convert the transformed question into vector
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: "text-embedding-004",
    });

    // Make sure we're passing a string to embedQuery
    const queryVector = await embeddings.embedQuery(String(transformedQuery));
    console.log("Embeddings created, querying Pinecone...");

    // Initialize Pinecone
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    const searchResults = await pineconeIndex.query({
      topK: 10,
      vector: queryVector,
      includeMetadata: true,
    });

    console.log("Pinecone query completed, generating response...");

    // Check if we got any results
    if (!searchResults.matches || searchResults.matches.length === 0) {
      console.log("No matching documents found in Pinecone");
      return;
    }

    // Create the context from search results
    const context = searchResults.matches
      .filter((match) => match.metadata && match.metadata.text)
      .map((match) => match.metadata.text)
      .join("\n\n--\n\n");

    if (!context.trim()) {
      console.log("No valid context found in search results");
      return;
    }

    console.log("Context created, calling AI...");

    // Create context from history for the main response
    const historyContext =
      History.length > 0
        ? `\n\nChat History:\n${History.map(
            (h) => `${h.role}: ${h.content}`
          ).join("\n")}`
        : "";

    // Create the full prompt with context and question
    const fullPrompt = `You have to behave like a data structure algorithm expert.
You will be given a context of relevant information and a user question.
Your task is to answer the user's question based only on the provided context.
If the answer is not in the context, you must say "I could not find the answer in the provided documents".
Keep your answer clear, concise and educational.

Context: ${context}${historyContext}

Original Question: ${question}
Transformed Question: ${transformedQuery}`;

    // Call the AI with the full prompt as a single user message
    const response = await ai.invoke([{ role: "user", content: fullPrompt }]);

    // Add to history after successful response
    History.push({
      role: "user",
      content: question,
    });

    History.push({
      role: "assistant",
      content: response.content,
    });

    console.log("\n=== AI Response ===");
    console.log(response.content);
    console.log("===================\n");
  } catch (error) {
    console.error("Error in chatting function:", error);
    console.error("Error details:", error.message);

    // More detailed error logging
    if (error.response) {
      console.error("API Response Error:", error.response.data);
    }
    if (error.code) {
      console.error("Error Code:", error.code);
    }
    if (error.stack) {
      console.error("Stack trace:", error.stack);
    }
  }
}

async function main() {
  console.log("RAG Chatbot initialized. Type 'exit' to quit.\n");

  while (true) {
    try {
      const userProblem = readlineSync.question("Ask me anything --> ");

      // Allow user to exit
      if (userProblem.toLowerCase() === "exit") {
        console.log("Goodbye!");
        break;
      }

      if (userProblem.trim()) {
        await chatting(userProblem);
      }
    } catch (error) {
      console.error("Error in main loop:", error);
      break;
    }
  }
}

// Add error handling for the main function
main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
