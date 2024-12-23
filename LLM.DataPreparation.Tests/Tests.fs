module LLM.DataPreparation.Tests

open FsUnit

open NUnit.Framework

[<Test>]
let ``build vocabulary`` () =

    // Test
    let vocabulary = DataSource.createVocabulary "Here's some content."

    // Verify
    vocabulary.Count |> should be (greaterThanOrEqualTo 4)

//[<Test>]
//let ``encode something`` () =

//    // Setup
//    let content    = "First of all, some text goes here."
//    let vocabulary = content |> DataSource.createVocabulary
//    let textInput  = "First of all"
    
//    vocabulary 
//    |> Tokenizer.encode textInput
//    |> Array.length
//    |> should be (greaterThan 3)

[<Test>]
let ``get input/target pairs`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let textInput  = "First of all"
    let elementsInRow = 4
    let stride = 1
    let contentTokens = vocabulary |> Tokenizer.extractTokens content

    // Test
    let input, target = Tokenizer.inputTargetPairs contentTokens elementsInRow stride

    input .[0].[0] |> should equal 0
    target.[0].[0] |> should equal 1

//[<Test>]
//let ``decode something`` () =

//    Tokenizer.decode [||]
//    |> should equal false
