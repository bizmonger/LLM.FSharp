module LLM.Transformer.Tests

open FsUnit

open NUnit.Framework

[<Test>]
let ``encode something`` () =
    
    Tokenizer.encode "some text goes here."
    |> should equal [||]

[<Test>]
let ``decode something`` () =

    Tokenizer.decode [||]
    |> should equal "?"

[<Test>]
let ``build vocabulary`` () =

    let result = DataSource.toVocabulary "some text goes here."
    result |> should equal "?"
