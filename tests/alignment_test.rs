// Memory alignment test to verify struct sizes match between CUDA C++ and Rust

#[cfg(test)]
mod alignment_tests {
    use super::super::*;

    #[test]
    fn test_command_size() {
        let size = std::mem::size_of::<Command>();
        println!("Command size: {} bytes", size);

        // Verify we're using the correct size
        // Based on CUDA analysis, this should be 48 bytes
        assert_eq!(size, 48, "Command struct must be 48 bytes to match CUDA");
    }

    #[test]
    fn test_command_alignment() {
        let align = std::mem::align_of::<Command>();
        println!("Command alignment: {} bytes", align);

        // Should be aligned to at least 4 bytes
        assert!(align >= 4, "Command must be at least 4-byte aligned");
    }

    #[test]
    fn test_command_field_offsets() {
        // Verify critical field offsets
        assert_eq!(offset_of!(Command, cmd_type), 0);
        assert_eq!(offset_of!(Command, id), 4);
        assert_eq!(offset_of!(Command, timestamp), 8);
        // The union starts at offset 16
        assert_eq!(offset_of!(Command, data_a), 16);
        assert_eq!(offset_of!(Command, data_b), 20);
        assert_eq!(offset_of!(Command, result), 24);
        assert_eq!(offset_of!(Command, batch_data), 28);
        assert_eq!(offset_of!(Command, batch_count), 36);
        // Padding at offset 40
        assert_eq!(offset_of!(Command, _padding), 40);
        assert_eq!(offset_of!(Command, result_code), 44);
    }

    #[test]
    fn test_command_queue_size() {
        let size = std::mem::size_of::<CommandQueueHost>();
        println!("CommandQueueHost size: {} bytes", size);

        // Based on CUDA CommandQueue calculation:
        // status: 4 bytes
        // commands[16]: 16 * 48 = 768 bytes
        // head: 4 bytes
        // tail: 4 bytes
        // commands_processed: 8 bytes
        // total_cycles: 8 bytes
        // idle_cycles: 8 bytes
        // current_strategy: 4 bytes
        // consecutive_commands: 4 bytes
        // consecutive_idle: 4 bytes
        // last_command_cycle: 8 bytes
        // avg_command_latency_cycles: 8 bytes
        // padding: 64 bytes
        // Total: 4 + 768 + 4 + 4 + 8 + 8 + 8 + 4 + 4 + 4 + 8 + 8 + 64 = 896 bytes

        assert_eq!(size, 896, "CommandQueueHost must be 896 bytes to match CUDA");
    }

    #[test]
    fn test_command_queue_alignment() {
        let align = std::mem::align_of::<CommandQueueHost>();
        println!("CommandQueueHost alignment: {} bytes", align);

        // Should be aligned to cache line boundary
        assert!(align >= 8, "CommandQueueHost should be at least 8-byte aligned");
    }
}

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
