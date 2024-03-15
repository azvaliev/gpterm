use core::str;
use futures_util::StreamExt;
use std::{
    env, fs,
    io::{self, Write},
    path::{self, Path},
    process,
};
use tempdir::TempDir;

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

/// only print in dev/debug mode
macro_rules! debug_println {
    ($($arg:tt)*) => (if ::std::cfg!(debug_assertions) { ::std::println!($($arg)*); })
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
            print!(
                "{}\n{}\n\n{}",
                SIGNUP_PROMPT, SIGNUP_LINK, ENTER_API_KEY_PROMPT
            );
            io::stdout().flush().unwrap();

            let key = rpassword::read_password().unwrap_or_else(|_| {
                println!("Could not read api key. Please try again later");
                process::exit(exitcode::USAGE);
            });

            if let Err(e) = save_openai_api_key(&local_app_folder, &key) {
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
        // Sometimes it will split up an individual line of JSON as two SSE events
        let mut partial_message = String::new();

        while let Some(Ok(raw_res)) = res_stream.next().await {
            // Multiple events can get recieved at once so we split those up
            let responses = str::from_utf8(&raw_res)
                .expect("recieved a string response")
                .split("\n\n")
                .filter(|r| r.len() > 6 && !r.contains("[DONE]"));

            for (idx, response) in responses.enumerate() {
                let mut response_data = response.replace("data: ", ""); // Remove the SSE prefix
                    
                if !partial_message.is_empty() && idx == 0 {
                    response_data = format!("{}{}", &partial_message, response_data);
                }

                let (chat_id, response) =
                    match serde_json::from_str::<CompletionResponse>(&response_data) {
                        Ok(mut d) => {
                            partial_message = String::new();
                            (d.id, d.choices.remove(0).delta)
                        },
                        Err(e) => {
                            partial_message += &response_data;
                            debug_println!("[WARN] Recieved incomplete response {:?}\n", e);
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
    let api_key_env_var = cuid2::create_id();

    env::set_var(TOKEN_VARIABLE, &api_key_env_var);
    assert_eq!(
        get_openai_api_key(&path::PathBuf::new()),
        Some(api_key_env_var)
    );
    env::remove_var(TOKEN_VARIABLE);
}

#[test]
fn get_some_openai_api_key_prefer_env_var() {
    let api_key_env_var = cuid2::create_id();

    let fs_token = cuid2::create_id();
    let tmp_dir = TempDir::new(&cuid2::create_id()).expect("can create temp folder for test");
    let tmp_token_file = Path::join(&tmp_dir.path(), TOKEN_FILE);
    fs::write(&tmp_token_file, &fs_token).expect("can write temp token file");

    env::set_var(TOKEN_VARIABLE, &api_key_env_var);
    assert_eq!(
        get_openai_api_key(&tmp_dir.into_path()),
        Some(api_key_env_var)
    );
    env::remove_var(TOKEN_VARIABLE);
}

#[test]
fn get_some_openai_api_key_from_fs() {
    let example_token = cuid2::create_id();

    let tmp_dir = TempDir::new(&cuid2::create_id()).expect("can create temp folder for test");
    let tmp_token_file = Path::join(&tmp_dir.path(), TOKEN_FILE);
    fs::write(&tmp_token_file, &example_token).expect("can write temp token file");

    assert_eq!(
        get_openai_api_key(&tmp_dir.into_path()),
        Some(example_token)
    );
}

#[test]
fn get_none_openai_api_key_no_folder() {
    assert_eq!(
        get_openai_api_key(&path::PathBuf::from("this_doesnt_exist")),
        None
    );
}

#[test]
fn get_none_openai_api_key_no_file() {
    let tmp_dir = TempDir::new(&cuid2::create_id()).expect("can create temporary folder");

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

#[test]
fn save_ok_openai_api_key() {
    let tmpdir = TempDir::new(&cuid2::create_id()).expect("can create temporary folder");
    let tmpdir_path = tmpdir.into_path();
    let api_key = cuid2::create_id();

    println!("{}", tmpdir_path.display());

    assert!(!save_openai_api_key(&tmpdir_path, &api_key).is_err());

    let written_api_key =
        fs::read_to_string(Path::join(&tmpdir_path, TOKEN_FILE)).expect("can read token file");
    assert_eq!(written_api_key, api_key);
}

#[test]
fn save_ok_openai_api_key_create_folder() {
    let api_key = cuid2::create_id();
    let tmpdir_path = path::PathBuf::from(&cuid2::create_id());

    assert!(!save_openai_api_key(&tmpdir_path, &api_key).is_err());

    let written_api_key =
        fs::read_to_string(Path::join(&tmpdir_path, TOKEN_FILE)).expect("can read token file");
    assert_eq!(written_api_key, api_key);

    fs::remove_dir_all(tmpdir_path).expect("can cleanup temp dir");
}

#[test]
fn save_err_openai_api_key_invalid_folder() {
    let api_key = cuid2::create_id();

    assert!(save_openai_api_key(&path::PathBuf::new(), &api_key).is_err());
}

fn save_openai_api_key(local_app_folder: &path::PathBuf, api_key: &str) -> Result<(), io::Error> {
    if !local_app_folder.exists() {
        fs::create_dir(&local_app_folder)?;
    };

    let path_to_token_file = Path::join(&local_app_folder, TOKEN_FILE);
    fs::write(path_to_token_file, api_key)?;

    return Ok(());
}

#[derive(Debug)]
enum CompletionError {
    RequestSerialize,
    Unauthorized,
    OutOfTokens,
    UnknownRequest,
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
    .map_err(|_| CompletionError::RequestSerialize)?;

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
            match e.status() {
                Some(StatusCode::UNAUTHORIZED) => CompletionError::Unauthorized,
                Some(StatusCode::TOO_MANY_REQUESTS) => CompletionError::OutOfTokens,
                _ => CompletionError::UnknownRequest,
            }
        })?
        .bytes_stream();

    Ok(res)
}
