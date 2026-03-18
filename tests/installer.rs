
use cudaclaw::installer;

#[tokio::test]
async fn test_installer() {
    // Set up a dummy .env file for the test
    std::fs::write(".env", "LLM_API_KEY=dummy_key").unwrap();

    let result = installer::bootstrap_hardware("Persistent Spreadsheet Recalculation").await;

    // In a real test, we would check the file system.
    // Here, we just check if the function returns successfully.
    assert!(result.is_ok());

    // Clean up the dummy .env file
    std::fs::remove_file(".env").unwrap();
}
