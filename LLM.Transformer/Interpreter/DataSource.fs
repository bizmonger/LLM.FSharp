namespace LLM.Transformer

open Microsoft.ML;
open LLM.DataPreparation.Language
open LLM.DataPreparation

module DataSource =

    [<CLIMutable>]
    type InputData = { Text: Text }

    [<CLIMutable>]
    type TokenizedData = { Tokens: TokenedText[] }

    let toVocabulary : Operations.Text.ToVocabulary =

        fun content ->

            let mlContext = MLContext()

            let inputData = 
                [ {Text = content} ]
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
            let vocabulary = Vocabulary()
            let mutable tokenId = 0

            for row in tokenizedData do
                for token in row.Tokens do
                    if not (vocabulary.ContainsValue(token)) then
                        vocabulary.[tokenId] <- token
                        tokenId <- tokenId + 1

            vocabulary