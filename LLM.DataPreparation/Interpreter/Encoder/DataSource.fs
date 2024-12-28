namespace LLM.DataPreparation

open System.Collections.Generic
open System.Text.RegularExpressions

module DataSource =

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

            tokens
            |> Seq.distinct
            |> Seq.iteri (fun index token -> tokenDict.Add(token, index))

            let sortedDict =
                tokenDict
                |> Seq.sortBy (fun kvp -> kvp.Key)
                |> Dictionary

            sortedDict

    let toTokenLookup (inputDict: Dictionary<'K, 'V>) =

        let swappedDict = Dictionary<'V, 'K>()

        for KeyValue(key, value) in inputDict do
            swappedDict.[value] <- key

        swappedDict