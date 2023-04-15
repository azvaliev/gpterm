use core::str;
use std::{
    io::{self, Write},
    process,
};
use futures_util::StreamExt;
use serde_json;

use bytes::Bytes;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

#[macro_use]
extern crate dotenv_codegen;

const OPENAI_COMPLETION_ENDPOINT: &'static str = "https://api.openai.com/v1/chat/completions";
const OPENAI_API_KEY: &'static str = dotenv!("OPENAI_API_KEY");

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
enum MessageRole {
    User,
    Assistant,
}

#[derive(Deserialize, Serialize, Debug)]
struct Message {
    #[serde(skip)]
    id: String,
    role: MessageRole,
    content: String,
}

#[tokio::main]
async fn main() {
    let mut conversation: Vec<Message> = Vec::new();

    println!("Enter a message...");
    loop {
        // Add a prompt for user to enter message
        let mut message = String::new();
        print!("> ");
        let _ = io::stdout().flush();

        // Read users message
        io::stdin()
            .read_line(&mut message)
            .expect("Read input from terminal");

        // Match command or create new message
        match message.as_str() {
            "exit\n" => process::exit(exitcode::OK),
            "reset\n" => {
                conversation.clear();
                println!("Cleared previous conversation");
                continue;
            }
            _ => {
                conversation.push(Message {
                    id: cuid2::create_id(),
                    role: MessageRole::User,
                    content: message,
                });
            }
        }


        // Get ChatGPT response as SSE stream
        let mut res_stream = get_completion(&conversation).await.unwrap();
        while let Some(Ok(raw_res)) = res_stream.next().await {
            // Multiple events can get recieved at once so we split those up
            let responses = str::from_utf8(&raw_res)
                .expect("recieved a string response")
                .split("\n\n")
                .filter(|r| r.len() > 6 && !r.contains("[DONE]"));

            for response in responses {
                let (chat_id, response) = match serde_json::from_str::<CompletionResponse>(&response[6..]) {
                    Ok(mut d) => (d.id, d.choices.remove(0).delta),
                    Err(e) => {
                        eprintln!("Failed to read response {:?} - {}", response, e);
                        break;
                    }
                };

                // The first SSE will created a new message, but further ones should add to existing message
                let mut message = match conversation.last() {
                    Some(Message{ id, .. }) if id == &chat_id => conversation.pop().expect("last message should exist"),
                    _ => Message{
                        id: chat_id,
                        role: response.role.unwrap_or(MessageRole::Assistant),
                        content: String::new(),
                    }
                };
                let content = response.content.unwrap_or(String::new());

                // Since we are using print!() and not println!() we should flush
                print!("{}", &content);
                let _ = io::stdout().flush();

                message.content += &content;
                conversation.push(message);
            }
        };

        // Spacing between messages to make conversation easier to read
        print!("\n\n");
    };
}

#[derive(Debug)]
enum CompletionError {
    RequestSerializeError,
    UnauthorizedError,
    OutOfTokensError,
    UnknownRequestError,
}

// This is essentially one piece of the response from ChatGPT
#[derive(Deserialize)]
struct CompletionDelta {
    content: Option<String>,
    role: Option<MessageRole>,
}

#[derive(Deserialize)]
struct CompletionChoice {
    delta: CompletionDelta,
    // finish_reason: Option<String>,
    // index: u32,
}
#[derive(Deserialize)]
struct CompletionResponse {
    id: String,
    // object: String,
    // created: u32,
    // model: String,
    choices: Vec<CompletionChoice>,
}

async fn get_completion(
    conversation: &Vec<Message>,
) -> Result<impl futures_core::Stream<Item = Result<Bytes, reqwest::Error>>, CompletionError> {
    // Request body for OpenAI completion
    #[derive(Serialize)]
    struct CompletionBody<'a> {
        model: &'a str,
        messages: &'a Vec<Message>,
        stream: bool,
    }

    let request_body = serde_json::to_string(&CompletionBody {
        model: "gpt-3.5-turbo",
        messages: conversation,
        stream: true,
    })
    .map_err(|_| CompletionError::RequestSerializeError)?;

    // Open the streaming connection and handle any bad responses
    let client = reqwest::Client::new();
    let res = client
        .post(OPENAI_COMPLETION_ENDPOINT)
        .header("Authorization", format!("Bearer {}", OPENAI_API_KEY))
        .header("Content-Type", "application/json")
        .body(request_body)
        .send()
        .await
        .map_err(|e| {
            return match e.status() {
                Some(StatusCode::UNAUTHORIZED) => CompletionError::UnauthorizedError,
                Some(StatusCode::TOO_MANY_REQUESTS) => CompletionError::OutOfTokensError,
                _ => CompletionError::UnknownRequestError,
            };
        })?
        .bytes_stream();

    return Ok(res);
}
