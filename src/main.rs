use core::str;
use futures_util::StreamExt;
use serde_json;
use tempdir::TempDir;
use std::{
    io::{self, Write},
    process, path::{Path, self}, env, fs,
};

use bytes::Bytes;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

const SIGNUP_PROMPT: &'static str = "This app requires an OpenAI API key.\nYou can sign up for an OpenAI account for free and get yours using the below link";
const SIGNUP_LINK: &'static str = "https://platform.openai.com/account/api-keys";
const ENTER_API_KEY_PROMPT: &'static str = "Please enter your OpenAI API Key:";

const OPENAI_COMPLETION_ENDPOINT: &'static str = "https://api.openai.com/v1/chat/completions";

const TOKEN_VARIABLE: &'static str = "OPENAI_API_TOKEN";
const APP_FOLDER: &'static str = ".gpterm";
const TOKEN_FILE: &'static str = "token";

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
    let local_app_folder = match home::home_dir() {
        Some(path) => Path::join(&path, APP_FOLDER),
        None => {
            eprintln!("Could not determine your home directory!");
            process::exit(exitcode::OSFILE);
        }
    };
    
    // Retrieve previously saved users API token or ask them to input it
    let api_key = match get_openai_api_key(&local_app_folder) {
        Some(key) => key,
        None => {
            print!("{}\n{}\n\n{}", SIGNUP_PROMPT, SIGNUP_LINK, ENTER_API_KEY_PROMPT);
            io::stdout().flush().unwrap();
            
            let key = rpassword::read_password().unwrap_or_else(|_| {
                println!("Could not read api key. Please try again later");
                process::exit(exitcode::USAGE);
            });

            if let Err(e) = save_openai_api_key(&key) {
                eprintln!("Failed to save api key to disk {}", e);
            };

            key
        }
    };
    
    let mut conversation: Vec<Message> = Vec::new();

    println!("Type your message - when finished, type ;; and press enter");
    loop {
        let mut message = String::new();

        while !message.ends_with(";;\n") {
            // Add a prompt for user to enter message
            print!("> ");
            let _ = io::stdout().flush();

            // Read users message
            io::stdin()
                .read_line(&mut message)
                .expect("Read input from terminal");

            // Check for commands
            match message.as_str() {
                "exit\n" => process::exit(exitcode::OK),
                "reset" => {
                    conversation.clear();
                    println!("Cleared previous conversation");
                    message = String::new();
                    continue;
                }
                _ => {}
            }
        }

        // Add message to conversation
        conversation.push(Message {
            id: cuid2::create_id(),
            role: MessageRole::User,
            content: String::from(message.trim_end_matches(";;\n")),
        });

        // Get ChatGPT response as SSE stream
        let mut res_stream = get_completion(&conversation, &api_key).await.unwrap();
        while let Some(Ok(raw_res)) = res_stream.next().await {
            // Multiple events can get recieved at once so we split those up
            let responses = str::from_utf8(&raw_res)
                .expect("recieved a string response")
                .split("\n\n")
                .filter(|r| r.len() > 6 && !r.contains("[DONE]"));

            for response in responses {
                let (chat_id, response) =
                    match serde_json::from_str::<CompletionResponse>(&response[6..]) {
                        Ok(mut d) => (d.id, d.choices.remove(0).delta),
                        Err(e) => {
                            eprintln!("Failed to read response {:?} - {}", response, e);
                            break;
                        }
                    };

                // The first SSE will created a new message, but further ones should add to existing message
                let mut message = match conversation.last() {
                    Some(Message { id, .. }) if id == &chat_id => {
                        conversation.pop().expect("last message should exist")
                    }
                    _ => Message {
                        id: chat_id,
                        role: response.role.unwrap_or(MessageRole::Assistant),
                        content: String::new(),
                    },
                };
                let content = response.content.unwrap_or(String::new());

                // Since we are using print!() and not println!() we should flush
                print!("{}", &content);
                let _ = io::stdout().flush();

                message.content += &content;
                conversation.push(message);
            }
        }

        // Spacing between messages to make conversation easier to read
        print!("\n\n");
    }
}

#[test]
fn get_some_openai_api_key_from_env_var() {
    let api_key_env_var = String::from("TE$1_T3ST");

    env::set_var(TOKEN_VARIABLE, &api_key_env_var);
    assert_eq!(get_openai_api_key(&path::PathBuf::new()), Some(api_key_env_var));
    env::remove_var(TOKEN_VARIABLE);
}

#[test]
fn get_some_openai_api_key_prefer_env_var() {
    let api_key_env_var = String::from("ENV_V@R_T3$T");

    let fs_token = String::from("F0o_B@r");
    let tmp_dir = TempDir::new("key_from_fs").expect("can create temp folder for test");
    let tmp_token_file = Path::join(&tmp_dir.path(), TOKEN_FILE);
    fs::write(&tmp_token_file, &fs_token).expect("can write temp token file");

    env::set_var(TOKEN_VARIABLE, &api_key_env_var);
    assert_eq!(get_openai_api_key(&tmp_dir.into_path()), Some(api_key_env_var));
    env::remove_var(TOKEN_VARIABLE);
}

#[test]
fn get_some_openai_api_key_from_fs() {
    let example_token = String::from("F0o_B@r");

    let tmp_dir = TempDir::new("key_from_fs").expect("can create temp folder for test");
    let tmp_token_file = Path::join(&tmp_dir.path(), TOKEN_FILE);
    fs::write(&tmp_token_file, &example_token).expect("can write temp token file");

    assert_eq!(get_openai_api_key(&tmp_dir.into_path()), Some(example_token));
}

#[test]
fn get_none_openai_api_key_no_folder() {
    assert_eq!(get_openai_api_key(&path::PathBuf::from("this_doesnt_exist")), None);
}

#[test]
fn get_none_openai_api_key_no_file() {
    let tmp_dir = TempDir::new("this_doesnt_have_any_files").expect("can create temporary folder");

    assert_eq!(get_openai_api_key(&tmp_dir.into_path()), None);
}

fn get_openai_api_key<'a>(local_app_folder: &path::PathBuf) -> Option<String> {
    if let Ok(token) = env::var(TOKEN_VARIABLE) {
        return Some(token);
    };
    
    let path_to_token_file = Path::join(&local_app_folder, TOKEN_FILE);
    if !path_to_token_file.exists() {
        return None;
    };

    if let Ok(token) = fs::read_to_string(path_to_token_file) {
        return Some(token);
    };

    return None;
}

fn save_openai_api_key(api_key: &str) -> Result<(), io::Error> {
    let local_app_folder = match home::home_dir() {
        Some(path) => Path::join(&path, APP_FOLDER),
        None => {
            eprintln!("Could not determine your home directory");
            process::exit(exitcode::OSFILE);
        }
    };
    
    if !local_app_folder.exists() {
        fs::create_dir(&local_app_folder)?;
    };

    let path_to_token_file = Path::join(&local_app_folder, TOKEN_FILE);
    fs::write(path_to_token_file, api_key)?;

    return Ok(());
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
    api_key: &str,
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
        .header("Authorization", format!("Bearer {}", api_key))
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
