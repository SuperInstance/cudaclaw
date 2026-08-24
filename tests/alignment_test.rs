// Memory alignment test to verify struct sizes match between CUDA C++ and Rust
// Self-contained — redefines the structs to verify layout matches CUDA C++

// Safe offset_of macro for packed structs
macro_rules! offset_of {
    ($ty:ty, $field:ident) => {{
        let uninit = std::mem::MaybeUninit::<$ty>::uninit();
        let ptr = uninit.as_ptr();
        unsafe {
            let field_ptr = std::ptr::addr_of!((*ptr).$field);
            (field_ptr as *const u8).offset_from(ptr as *const u8) as usize
        }
    }};
}

#[repr(C, packed(8))]
#[derive(Debug, Clone, Copy)]
struct Command {
    pub cmd_type: u32,
    pub id: u32,
    pub timestamp: u64,
    pub data_a: f32,
    pub data_b: f32,
    pub result: f32,
    pub batch_data: u64,
    pub batch_count: u32,
    pub _padding: u32,
    pub result_code: u32,
}

const QUEUE_SIZE: usize = 1024;

#[repr(C, packed(8))]
#[derive(Clone, Copy)]
struct CommandQueueHost {
    pub buffer: [Command; QUEUE_SIZE],
    pub status: u32,
    pub head: u32,
    pub tail: u32,
    pub is_running: bool,
    pub _padding: [u8; 3],
    pub commands_sent: u64,
    pub commands_processed: u64,
    pub _stats_padding: [u8; 8],
}

#[cfg(test)]
mod alignment_tests {
    use super::*;

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
        assert_eq!(size, 49192, "CommandQueueHost must be 49192 bytes to match CUDA");
    }

    #[test]
    fn test_command_queue_alignment() {
        let align = std::mem::align_of::<CommandQueueHost>();
        println!("CommandQueueHost alignment: {} bytes", align);
        assert!(align >= 4, "CommandQueueHost should be at least 4-byte aligned");
    }
}
