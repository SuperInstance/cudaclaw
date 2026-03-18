// Memory alignment test to verify struct sizes match between CUDA C++ and Rust
// These tests require the `cuda` feature because they reference CUDA-dependent types
// (Command, CommandQueueHost) which are only available when CUDA is enabled.
//
// Run with: cargo test --features cuda --test alignment_test

// This entire test file requires CUDA types that are only available with the cuda feature.
// Without the feature, there are no types to test alignment on.
#[cfg(feature = "cuda")]
mod alignment_tests {
    use cudaclaw::{Command, CommandQueueHost};

    // Helper macro to get field offset
    macro_rules! offset_of {
        ($ty:ty, $field:ident) => {{
            let uninit = std::mem::MaybeUninit::<$ty>::uninit();
            let ptr = uninit.as_ptr();
            unsafe {
                (&(*ptr).$field as *const _ as usize) - (ptr as usize)
            }
        }};
    }

    #[test]
    fn test_command_size() {
        let size = std::mem::size_of::<Command>();
        println!("Command size: {} bytes", size);
        assert_eq!(size, 48, "Command struct must be 48 bytes to match CUDA");
    }

    #[test]
    fn test_command_alignment() {
        let align = std::mem::align_of::<Command>();
        println!("Command alignment: {} bytes", align);
        assert!(align >= 4, "Command must be at least 4-byte aligned");
    }

    #[test]
    fn test_command_field_offsets() {
        assert_eq!(offset_of!(Command, cmd_type), 0);
        assert_eq!(offset_of!(Command, id), 4);
        assert_eq!(offset_of!(Command, timestamp), 8);
        assert_eq!(offset_of!(Command, data_a), 16);
        assert_eq!(offset_of!(Command, data_b), 20);
        assert_eq!(offset_of!(Command, result), 24);
        assert_eq!(offset_of!(Command, batch_data), 28);
        assert_eq!(offset_of!(Command, batch_count), 36);
        assert_eq!(offset_of!(Command, _padding), 40);
        assert_eq!(offset_of!(Command, result_code), 44);
    }

    #[test]
    fn test_command_queue_size() {
        let size = std::mem::size_of::<CommandQueueHost>();
        println!("CommandQueueHost size: {} bytes", size);
        assert_eq!(size, 896, "CommandQueueHost must be 896 bytes to match CUDA");
    }

    #[test]
    fn test_command_queue_alignment() {
        let align = std::mem::align_of::<CommandQueueHost>();
        println!("CommandQueueHost alignment: {} bytes", align);
        assert!(align >= 8, "CommandQueueHost should be at least 8-byte aligned");
    }
}
