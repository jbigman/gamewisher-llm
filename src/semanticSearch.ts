// Import necessary libraries
import * as use from '@tensorflow-models/universal-sentence-encoder'
import '@tensorflow/tfjs'  // Ensure TensorFlow.js is loaded
import { similarity } from 'ml-distance'
import { input } from '@inquirer/prompts'

// Load the Universal Sentence Encoder model de tensorflow
async function loadUSEModel(): Promise<use.UniversalSentenceEncoder> {
  const model = await use.load()
  return model
}

// Function to compute embeddings for an array of texts
async function computeEmbeddings(texts: string[]): Promise<number[][]> {
  const model = await loadUSEModel()
  const embeddings = await model.embed(texts)
  const embeddingsArray: number[][] = await embeddings.array() as number[][]  // Convert to array
  return embeddingsArray
}

// Custom VectorStore Class
class EmbeddingVectorStore {
  private texts: string[]
  private vectors: number[][]

  constructor(texts: string[], vectors: number[][]) {
    this.texts = texts
    this.vectors = vectors
  }

  // Method to compute cosine similarity between two vectors
  private computeCosineSimilarity(vec1: number[], vec2: number[]): number {
    return similarity.cosine(vec1, vec2)
  }

  // Perform a similarity search for a query embedding
  public similaritySearch(queryEmbedding: number[], topN = 3): { text: string, score: number }[] {
    // Compute similarities
    const similarities = this.vectors.map((vector, index) => ({
      text: this.texts[index],
      score: this.computeCosineSimilarity(queryEmbedding, vector),
    }))

    // Sort by similarity score (descending) and return top N results
    return similarities.sort((a, b) => b.score - a.score).slice(0, topN)
  }
}

// Function to perform similarity search using the custom VectorStore
async function performSearch(texts: string[], query: string, topNumber = 3): Promise<{ text: string, score: number }[]> {
  
  // Applique le model à tous les textes
  const embeddingsArray: number[][] = await computeEmbeddings(texts)

  // Initialize custom vector store with texts and their embeddings
  const vectorStore = new EmbeddingVectorStore(texts, embeddingsArray)

  // Applique le model à la requête
  const queryEmbedding: number[] = (await computeEmbeddings([query]))[0]

  // Perform similarity search
  const searchResults = vectorStore.similaritySearch(queryEmbedding, topNumber)
  return searchResults
}
// Function to handle the interactive prompt loop
async function interactiveSearch() {
  const texts = [
    "j'aime les frites",
    "je deteste les jeux videos",
    "je joue aux jeux videos de temps en temps",
    "les jeux de cartes sont les meilleur",
    "la couture c'est mon dada"
  ]

  while (true) {
    // Use prompt to get user input
    const query = await input({message: "Entrez une phrase pour rechercher la similarité (ou tapez 'exit' pour quitter):"})

    // Exit condition
    if (!query) {
      break
    }

    // Exit condition
    if (query?.toLowerCase() === 'exit') {
      console.log("Exiting...")
      break
    }

    try {
      const results = await performSearch(texts, query, 3)
      console.log("Résultats")
      results.forEach((result, index) => {
        console.log(`Rank ${index + 1}:`, result.text, `(Score: ${result.score.toFixed(4)})`)
      })
    } catch (error) {
      console.error("Error performing search:", error)
    }
  }
}

// Start the interactive prompt loop
interactiveSearch()