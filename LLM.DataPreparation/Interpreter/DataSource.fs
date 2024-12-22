namespace LLM.DataPreparation

open System.Collections.Generic
open Language

module DataSource =

    let private getNumberOfIterations (text: string) =
        match text.Length with
        | len when len < 50 -> 10
        | len when len < 100 -> 20
        | len when len < 1_000 -> 30
        | len when len < 10_000 -> 40
        | len when len < 100_000 -> 50
        | len when len < 1_000_000 -> 60
        | len when len < 1_000_000_000 -> 70
        | _ -> 200

    let createVocabulary : Operations.Text.ToVocabulary =

        fun text ->

            // Step 1: Tokenize the text into single characters initially
            let mutable tokens: string list = 
                text
                |> Seq.toList
                |> List.map string

            let mutable corpus = Dictionary<string, int>()

            // Helper to update corpus frequencies for token pairs
            let updateCorpus tokens =
                corpus.Clear()
                tokens
                |> List.windowed 2 // Create overlapping pairs of two tokens
                |> List.iter (fun pair ->
                    let key = pair.[0] + pair.[1]
                    if corpus.ContainsKey(key) then
                        corpus.[key] <- corpus.[key] + 1
                    else
                        corpus.[key] <- 1)

            // Initialize the corpus
            updateCorpus tokens

            let numIterations = getNumberOfIterations text

            // Step 2: Perform Byte Pair Encoding for the specified number of iterations
            let mutable continueProcessing = true
            for _ in 1 .. numIterations do
                if not continueProcessing then
                    () // Exit loop early if no pairs to merge
                else
                    if corpus.Count = 0 then
                        continueProcessing <- false // Stop processing if no pairs exist
                    else
                        // Find the most frequent pair
                        let mostFrequentPair =
                            corpus
                            |> Seq.maxBy (fun kvp -> kvp.Value)
                            |> fun kvp -> kvp.Key

                        // Merge the most frequent pair in the token list
                        let newTokens = ResizeArray<string>()
                        let mutable i = 0
                        while i < tokens.Length do
                            if i < tokens.Length - 1 && (tokens.[i] + tokens.[i + 1]) = mostFrequentPair then
                                newTokens.Add(mostFrequentPair) // Merge the pair
                                i <- i + 2 // Skip the merged pair
                            else
                                newTokens.Add(tokens.[i])
                                i <- i + 1
                        tokens <- List.ofSeq newTokens

                        // Update the corpus after merging
                        updateCorpus tokens

            // Step 3: Create a final vocabulary with unique token IDs
            let vocabulary = Dictionary<string, int>()
            let mutable tokenId = 0

            tokens
            |> List.distinct // Ensure each token is unique in the vocabulary
            |> List.iter (fun token ->
                if not (vocabulary.ContainsKey(token)) then
                    vocabulary.[token] <- tokenId
                    tokenId <- tokenId + 1)

            vocabulary

