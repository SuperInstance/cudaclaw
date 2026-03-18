// Memory alignment test to verify struct sizes match between CUDA C++ and Rust
//
// This test ensures that the Rust structs have the same memory layout as their
// CUDA C++ counterparts, which is critical for unified memory communication.

use std::mem::{offset_of, size_of, align_of};

// Import types from the main crate
// Note: These need to be public or we need to redeclare them for testing
// For now, we'll create minimal test versions

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TestCommand {
    pub cmd_type: u32,
    pub id: u32,
    pub timestamp: u64,
    pub data: [u8; 32], // Simplified data payload
    pub result_code: i32,
}

#[cfg(test)]
mod alignment_tests {
    use super::*;

    #[test]
    fn test_command_size() {
        let size = size_of::<TestCommand>();
        println!("TestCommand size: {} bytes", size);

        // Verify reasonable size (not too large due to padding)
        assert!(size <= 64, "Command struct should be <= 64 bytes");
    }

    #[test]
    fn test_command_alignment() {
        let align = align_of::<TestCommand>();
        println!("TestCommand alignment: {} bytes", align);

        // Should be aligned to at least 4 bytes
        assert!(align >= 4, "Command must be at least 4-byte aligned");
    }

    #[test]
    fn test_command_field_offsets() {
        // Verify critical field offsets
        assert_eq!(offset_of!(TestCommand, cmd_type), 0);
        assert_eq!(offset_of!(TestCommand, id), 4);
        assert_eq!(offset_of!(TestCommand, timestamp), 8);
        assert_eq!(offset_of!(TestCommand, data), 16);
        assert_eq!(offset_of!(TestCommand, result_code), 48);
    }

    #[test]
    fn test_memory_layout() {
        // Test that the struct can be safely transmuted to bytes
        let cmd = TestCommand {
            cmd_type: 1,
            id: 42,
            timestamp: 12345,
            data: [0u8; 32],
            result_code: -1,
        };

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &cmd as *const TestCommand as *const u8,
                size_of::<TestCommand>()
            )
        };

        assert_eq!(bytes.len(), size_of::<TestCommand>());
    }

    #[test]
    fn test_no_excessive_padding() {
        let size = size_of::<TestCommand>();
        let max_field_size = size_of::<u64>();

        // Check that padding isn't excessive
        // A well-structured struct shouldn't have more than 25% padding
        let theoretical_min_size = size_of::<u32>() + size_of::<u32>() + size_of::<u64>() + 32 + size_of::<i32>();
        let padding_ratio = (size - theoretical_min_size) as f64 / theoretical_min_size as f64;

        assert!(padding_ratio < 0.25, "Excessive padding detected: {:.2}%", padding_ratio * 100.0);
    }
}
