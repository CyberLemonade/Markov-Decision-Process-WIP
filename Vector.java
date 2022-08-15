public class Vector extends Matrix
{
    // CONSTRUCTOR:
    
    Vector(final int N)
    {
        super(N, 1);
    }
    
    Vector(final float[] A)
    {
        super(A.length, 1);
        for (int i = 0; i < R; i++)
        {
            M[i][0] = A[i];
        }
    }
    
    // FUNCTIONS:
    
    float[] toArray()
    {
        return this.transpose().M[0].clone();
    }
    
    float dotProduct(Vector B)
    {
        Vector product = (Vector)Matrix.multiply(this, B);
        float dot = 0.0f;
        for (int i = 0; i < R; i++)
        {
            dot += product.M[i][0];
        }
        return dot;
    }
    
    @Override
    public String toString()
    {
        String debug = "";
        for (int i = 0; i < R; i++) 
        {
            for (int j = 0; j < C; j++) 
            {
                debug += M[i][j] + " ";
            }
            debug += "\n";
        }
        return debug;
    }
    
    // MOST FUNCTIONS WERE NOT EXPLICITLY MODIFIED AS A SIMPLE TYPECAST TO VECTOR WORKS
}