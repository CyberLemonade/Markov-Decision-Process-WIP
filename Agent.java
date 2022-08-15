import java.util.ArrayList;
import java.util.Collections;
import java.util.Arrays;

public abstract class Agent
{
    final int ACTION_SIZE;
    final int STATE_SIZE;
    
    final float GAMMA = 0.99f;
    final float EXPLORATION_PROB_DECAY = 0.05f;
    float EXPLORATION_PROB = 1.0f;
    
    ArrayList<Memory> replay_buffer;
    final int MAX_REPLAY_BUFFER = 2000;
    
    NeuralNetwork q_table;
    final int BATCH_SIZE = 32;
    
    Agent(final int ACTION_SIZE, final int STATE_SIZE)
    {
        this.ACTION_SIZE = ACTION_SIZE;
        this.STATE_SIZE = STATE_SIZE;
        
        this.replay_buffer = new ArrayList<Memory>(MAX_REPLAY_BUFFER + 100);
        
        int[] NN_layers = {STATE_SIZE, 24, 24, ACTION_SIZE};
        this.q_table = new NeuralNetwork(NN_layers);
    }
    
    int get_action(final float[] current_state)
    {
        if (Math.random() < EXPLORATION_PROB)
        {
            return (int)(Math.random() * ACTION_SIZE);
        }
        else
        {
            float[] q_values = q_table.guess(current_state);
            int best_action = 0;
            for (int i = 1; i < q_values.length; i++)
            {
                if (q_values[i] > q_values[best_action])
                {
                    best_action = i;
                }
            }
            return best_action;
        }
    }
    
    void update_EXPLORATION_PROB()
    {
        EXPLORATION_PROB *= Math.exp(-EXPLORATION_PROB_DECAY);
    }
    
    void add_replay(final float[] current_state, final int action, final int reward, final float[] next_state)
    {
        replay_buffer.add(new Memory(current_state, action, reward, next_state));
        if (replay_buffer.size() > MAX_REPLAY_BUFFER)
        {
            replay_buffer.remove(0);
        }
    }
    
    void train()
    {
        Collections.shuffle(replay_buffer);
        
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            Memory current_episode = replay_buffer.get(i);
            
            float[] q_values_predicted = q_table.guess(current_episode.current_state);
            
            float[] q_values_next_state = q_table.guess(current_episode.next_state);
            
            int best_action = 0;
            for (int j = 1; j < q_values_next_state.length; j++)
            {
                if (q_values_next_state[j] > q_values_next_state[best_action])
                {
                    best_action = j;
                }
            }
            
            float q_value_expected = current_episode.reward + GAMMA * q_values_next_state[best_action];
            
            float[] q_values_expected  = q_values_predicted.clone();
            q_values_expected[current_episode.action] = q_value_expected;
            
            q_table.train(current_episode.current_state, q_values_expected);
        }
    }
}