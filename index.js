import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import * as dotenv from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
dotenv.config()

async function indexDocumet(){
    const PDF_PATH = './Dsa.pdf' ;
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load()
    console.log("rawDocs completed")
    // console.log(rawDocs.length)

    //chunking
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    })

    const chunkedDocs= await textSplitter.splitDocuments(rawDocs)
    console.log("chunkedDocs")

    //vector embedding

    const embedding = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model:'text-embedding-004'
    })

    console.log("embedding model created")

    //data base config
    //initialize pinecone client
    console.log(process.env.PINECONE_API_KEY)
    const pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    });
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME)

    //langchain (chunking, embedding, database)
    await PineconeStore.fromDocuments(chunkedDocs, embedding, {
        pineconeIndex: pineconeIndex,
        maxConcurrency: 5
    })

    console.log("stored in db")
}
indexDocumet()