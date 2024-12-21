﻿namespace LLM.Transformer

open System.Collections.Generic
open Microsoft.ML;
open LLM.DataPreparation.Language

module DataSource =

    [<CLIMutable>]
    type InputData = { Text: Text }

    [<CLIMutable>]
    type TokenizedData = { Tokens: TokenedText[] }

    let toVocabulary (text:string) : Dictionary<Token,TokenedText> =

        let mlContext = MLContext()

        let inputData = 
            [ {Text= text} ]
            |> mlContext.Data.LoadFromEnumerable

        // Define text processing pipeline
        let textPipeline = 
            mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text")

        // Fit and transform data
        let transformer = textPipeline.Fit(inputData)
        let transformedData = transformer.Transform(inputData)

        // Extract tokens from transformed data
        let tokenizedData = 
            mlContext.Data.CreateEnumerable<TokenizedData>(transformedData, reuseRowObject = false)
            |> Seq.toList

        // Create dictionary of token IDs
        let tokenDictionary = Dictionary<Token,TokenedText>()
        let mutable tokenId = 0

        for row in tokenizedData do
            for token in row.Tokens do
                if not (tokenDictionary.ContainsValue(token)) then
                    tokenDictionary.[tokenId] <- token
                    tokenId <- tokenId + 1

        tokenDictionary