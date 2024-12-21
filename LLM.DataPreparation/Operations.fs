namespace LLM.DataPreparation

open Language

module Operations =

    module Text =

        type ToTokenizedText = Text -> TokenedText
        type ToTokenMap      = Text -> (TokenedText * Tokens)
        type ToTokens        = Text -> Tokens
        type ToVocabulary    = Text -> Vocabulary
        type ToEmbedding     = Text -> Vectors

    module TokenedText = type ToTokens     = TokenedText -> Tokens
    module Token       = type toText       = Token       -> Text

    module Vectors =

        type PredictToken = Vectors -> Prediction
        type Add = TokenEmbedding -> PositionalEmbedding -> InputEmbedding

    module Tokens = 
    
        type ToTokenEmbedding      = Tokens -> Vocabulary -> DimensionCount -> TokenEmbedding
        type ToPositionalEmbedding = Tokens -> Vocabulary -> DimensionCount -> PositionalEmbedding
        type ToInputEmbedding      = Tokens -> Vocabulary -> DimensionCount -> InputEmbedding

    module Tokenizer =

        type Encode = Text.ToEmbedding
        type Decode = Vectors.PredictToken

    module DataLoader =

        module Get =

            type InputTargetPair = Vocabulary -> InputTokens -> Stride -> InputTokens * TargetTokens
            type TokenTensor     = Vocabulary -> Text -> BatchSize -> MaxRowSize -> Vectors
    
    module Get =

        open DataLoader

        module Tokens =

            type InputTargetPair = Get.InputTargetPair

        module Vector =

            type Size     = Vector -> int
            type Elements = Vector -> int

        module Vocabulary =

            type Size  = Vocabulary -> int
            type Items = Vocabulary -> (int * TokenedText) seq