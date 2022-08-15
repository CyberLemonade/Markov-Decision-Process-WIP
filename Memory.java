public class Memory
{
    float[] current_state;
    int action;
    float reward;
    float[] next_state;
    
    Memory(float[] current_state, int action, float reward, float[] next_state)
    {
        this.current_state = current_state.clone();
        this.action = action;
        this.reward = reward;
        this.next_state = next_state.clone();
    }
}