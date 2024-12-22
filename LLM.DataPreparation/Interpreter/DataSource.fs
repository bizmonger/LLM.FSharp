namespace LLM.DataPreparation

open System.Linq
open System.Collections.Generic

module DataSource =

    /// Function to merge the most frequent pair in the corpus
    let private mergePair (pair: string) (corpus: Dictionary<string, int>) : Dictionary<string, int> =

        let newCorpus = Dictionary<string, int>()

        for KeyValue(k, v) in corpus do

            // Replace the pair in the current token
            let newK = k.Replace(pair, string(pair.[0]) + string(pair.[1]))
            if newCorpus.ContainsKey(newK) then
                newCorpus.[newK] <- newCorpus.[newK] + v
            else
                newCorpus.[newK] <- v
        newCorpus

    let private getNumberOfIterations (text:string) =
        match text with
        | text when text.Length < 50            -> 1
        | text when text.Length < 100           -> 5
        | text when text.Length < 1_000         -> 10
        | text when text.Length < 10_000        -> 20
        | text when text.Length < 100_000       -> 30
        | text when text.Length < 1_000_000     -> 50
        | text when text.Length < 1_000_000_000 -> 100
        | _ -> 200

    /// Function to tokenize input text using byte-pair encoding
    let createVocabulary : Operations.Text.ToVocabulary =

        fun text ->

            let numIterations = getNumberOfIterations text

            // Step 1: Initial tokenization by treating each character as a separate token
            let tokens = 
                text 
                |> Seq.toList 
                |> List.map (fun c -> string c)
    
            let mutable corpus = Dictionary<string, int>()
    
            // Create initial frequencies
            for i in 0 .. tokens.Length - 2 do
                let pair = tokens.[i] + tokens.[i + 1]
                if corpus.ContainsKey(pair) then
                    corpus.[pair] <- corpus.[pair] + 1
                else
                    corpus.[pair] <- 1
    
            // Step 2: Perform Byte Pair Encoding for the specified number of iterations
            for _ in 1 .. numIterations do
                // Step 2a: Find the most frequent pair
                let maxPair = 
                    corpus 
                    |> Seq.maxBy (fun kvp -> kvp.Value)
                    |> fun kvp -> kvp.Key

                // Step 2b: Merge the most frequent pair
                corpus <- mergePair maxPair corpus
        
                // Step 2c: Update the corpus after merging the pair
                let newCorpus = Dictionary<string, int>()
                for KeyValue(k, v) in corpus do
                    if newCorpus.ContainsKey(k) then
                        newCorpus.[k] <- newCorpus.[k] + v
                    else
                        newCorpus.[k] <- v
                corpus <- newCorpus

            // Step 3: Create a final vocabulary with unique token IDs
            let vocabulary = Dictionary<string, int>()
            let mutable tokenId = 0

            // Assign unique token IDs to each token in the corpus
            for KeyValue(token, _) in corpus do
                vocabulary.[token] <- tokenId
                tokenId <- tokenId + 1

            let sorted = vocabulary
                        |> Seq.sortBy (fun kvp -> kvp.Key)
                        |> Dictionary
            sorted
