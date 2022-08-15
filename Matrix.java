public class Matrix
{
    final int R, C;
    float[][] M;
    
    // CONSTRUCTORS:
    
    Matrix(final int R, final int C)
    {
        this.R = R;
        this.C = C;
        this.M = new float[R][C];
    }
    
    Matrix(final int R, final int C, final float MIN, final float MAX)
    {
        this(R, C);
        
        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                M[i][j] = (float) Math.random() * (MAX - MIN) + MIN;
            }
        }
    }
    
    // FUNCTIONS:
    
    void add(final float VAL)
    {
        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                M[i][j] += VAL;
            }
        }        
    }
    
    void scale(final float VAL)
    {
        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                M[i][j] *= VAL;
            }
        }        
    }    
    
    void add(final Matrix B)
    {
        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                M[i][j] += B.M[i][j];
            }
        }
    }
    
    void multiply(final Matrix B)
    {
        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                M[i][j] *= B.M[i][j];
            }
        }
    }
    
    Matrix transpose()
    {
        Matrix T = new Matrix(C, R);
        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                T.M[j][i] = this.M[i][j];
            }
        }
        return T;
    }
    
    void sigmoid()
    {
        // s(x) = 1/1+e^{-x}
        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                M[j][i] = 1.0f / (1.0f + (float)Math.exp(-M[i][j]));
            }
        }
    }
    
    void dsigmoid()
    {
        // s'(x) = s(x) * (1 - s(x))
        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                M[j][i] = M[i][j] * (1.0f - M[i][j]);
            }
        }
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
    
    public Matrix clone()
    {
        Matrix CLONE = new Matrix(R, C);
        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                CLONE.M[j][i] = this.M[i][j];
            }
        }
        return CLONE;
    }
    
    // STATIC FUNCTIONS:
    
    static Matrix product(final Matrix A, final Matrix B)
    {
        if (A.C != B.R)
        {
            return null;
        }
        else
        {
            Matrix C = new Matrix(A.R, B.C);
            for (int i = 0; i < C.R; i++) 
            {
                for (int j = 0; j < C.C; j++) 
                {
                    float sum = 0.0f;
                    for (int k = 0; k < A.C; k++) 
                    {
                        sum += A.M[i][k] * B.M[k][j];
                    }
                    C.M[i][j] = sum;
                }
            }
            return C;
        }
    }
    
    static Matrix add(final Matrix A, final float VAL)
    {
        Matrix B = new Matrix(A.R, A.C);
        for (int i = 0; i < B.R; i++)
        {
            for (int j = 0; j < B.C; j++)
            {
                B.M[i][j] = A.M[i][j] + VAL;
            }
        }
        return B;
    }
    
    static Matrix scale(final Matrix A, final float VAL)
    {
        Matrix B = new Matrix(A.R, A.C);
        for (int i = 0; i < B.R; i++)
        {
            for (int j = 0; j < B.C; j++)
            {
                B.M[i][j] = A.M[i][j] * VAL;
            }
        }
        return B;    
    }    
    
    static Matrix add(final Matrix A, final Matrix B)
    {
        Matrix C = new Matrix(A.R, A.C);
        for (int i = 0; i < C.R; i++)
        {
            for (int j = 0; j < C.C; j++)
            {
                C.M[i][j] = A.M[i][j] + B.M[i][j];
            }
        }
        return C;
    }
    
    static Matrix multiply(final Matrix A, final Matrix B)
    {
        Matrix C = new Matrix(A.R, A.C);
        for (int i = 0; i < C.R; i++)
        {
            for (int j = 0; j < C.C; j++)
            {
                C.M[i][j] = A.M[i][j] * B.M[i][j];
            }
        }
        return C;
    }
    
    static Matrix transpose(final Matrix A)
    {
        Matrix T = new Matrix(A.C, A.R);
        for (int i = 0; i < A.R; i++)
        {
            for (int j = 0; j < A.C; j++)
            {
                T.M[j][i] = A.M[i][j];
            }
        }
        return T;
    }
    
    static Matrix sigmoid(final Matrix A)
    {
        // s(x) = 1/1+e^{-x}
        Matrix S = new Matrix(A.R, A.C);
        for (int i = 0; i < S.R; i++)
        {
            for (int j = 0; j < S.C; j++)
            {
                S.M[j][i] = 1.0f / (1.0f + (float)Math.exp(-A.M[i][j]));
            }
        }
        return S;
    }
    
    static Matrix dsigmoid(final Matrix A)
    {
        // s'(x) = s(x) * (1 - s(x))
        Matrix S = new Matrix(A.R, A.C);
        for (int i = 0; i < S.R; i++)
        {
            for (int j = 0; j < S.C; j++)
            {
                S.M[j][i] = A.M[i][j] * (1.0f - A.M[i][j]);
            }
        }
        return S;
    }
    
    static Matrix addRow(final Matrix A)
    {
        Matrix AR = new Matrix(A.R + 1, A.C);
        for (int i = 0; i < A.R; i++)
        {
            for (int j = 0; j < A.C; j++)
            {
                AR.M[j][i] = A.M[i][j];
            }
        }
        for (int j = 0; j < A.C; j++)
        {
            AR.M[A.R][j] = 1.0f;
        }
        return AR;
    }
    
    static Matrix addCol(final Matrix A)
    {
        Matrix AC = new Matrix(A.R, A.C + 1);
        for (int i = 0; i < A.R; i++)
        {
            for (int j = 0; j < A.C; j++)
            {
                AC.M[j][i] = A.M[i][j];
            }
        }
        for (int j = 0; j < A.R; j++)
        {
            AC.M[j][A.C] = 1.0f;
        }
        return AC;
    }
    
    static Matrix delRow(final Matrix A)
    {
        Matrix AR = new Matrix(A.R - 1, A.C);
        for (int i = 0; i < A.R - 1; i++)
        {
            for (int j = 0; j < A.C; j++)
            {
                AR.M[j][i] = A.M[i][j];
            }
        }
        return AR;
    }
    
    static Matrix delCol(final Matrix A)
    {
        Matrix AC = new Matrix(A.R, A.C - 1);
        for (int i = 0; i < A.R; i++)
        {
            for (int j = 0; j < A.C - 1; j++)
            {
                AC.M[j][i] = A.M[i][j];
            }
        }
        return AC;
    }
}