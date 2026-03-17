#include "shared_types.h"
#include <stdio.h>

int main() {
    printf("Command size: %zu bytes\n", sizeof(Command));
    printf("CommandQueue size: %zu bytes\n", sizeof(CommandQueue));
    printf("Command::cmd_type offset: %zu\n", offsetof(Command, cmd_type));
    printf("CommandQueue::buffer offset: %zu\n", offsetof(CommandQueue, buffer));
    printf("CommandQueue::status offset: %zu\n", offsetof(CommandQueue, status));
    printf("CommandQueue::head offset: %zu\n", offsetof(CommandQueue, head));
    printf("CommandQueue::tail offset: %zu\n", offsetof(CommandQueue, tail));
    printf("CommandQueue::is_running offset: %zu\n", offsetof(CommandQueue, is_running));
    printf("CommandQueue::commands_sent offset: %zu\n", offsetof(CommandQueue, commands_sent));
    printf("CommandQueue::commands_processed offset: %zu\n", offsetof(CommandQueue, commands_processed));
    printf("\nExpected buffer size: 48 * 1024 = %zu\n", 48 * 1024);
    printf("Actual buffer size (from status offset): %zu\n", 49152);
    printf("Difference: %zu bytes\n", 49152 - (48 * 1024));
    return 0;
}
