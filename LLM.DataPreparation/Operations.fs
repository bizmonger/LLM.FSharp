namespace LLM.DataPreparation

open Language

module Operations =

    module Text =

        type ToTokenizedText = Text -> TokenedText
        type ToTokenMap      = Text -> (TokenedText * Tokens)
        type ToTokens        = Text -> Tokens
        type ToVocabulary    = Text -> Vocabulary
        type ToEmbedding     = Text -> Vectors

    module TokenedText =

        type ToTokens = TokenedText -> Tokens

    module Token =

        type toText = Token -> Text

    module Vectors =

        type NextTokenPrediction = Vectors -> Prediction

    module List =

        module Vector =

            type Size     = Vector -> int
            type Elements = Vector -> int

        module Vocabulary =

            type Size  = Vocabulary -> int
            type Items = Vocabulary -> (int * TokenedText) seq
