namespace LLM.DataPreparation

open System.Collections.Generic
open System.Text.RegularExpressions
open Language

module DataSource =

    /// Function to tokenize text into words using a regular expression
    let private tokenize (text: string) =
        // Split text into tokens using non-word characters as delimiters
        let pattern = "\\w+|[^\\w\\s]+"
        Regex.Matches(text, pattern)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Value)

    let createVocabulary : Operations.Text.ToVocabulary =

        fun content ->

            let tokens = tokenize content |> Seq.toList
            let tokenDict = Dictionary<string, int>()

            // Populate the dictionary with tokens and their IDs
            tokens
            |> Seq.distinct
            |> Seq.iteri (fun index token -> tokenDict.Add(token, index))

            // Create a sorted dictionary by keys (tokens)
            let sortedDict =
                tokenDict
                |> Seq.sortBy (fun kvp -> kvp.Key)
                |> Dictionary

            sortedDict