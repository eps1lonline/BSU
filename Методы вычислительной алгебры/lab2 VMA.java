import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double[][] A = {{0.3857, -0.0508, 0.0102, 0.0203, 0.0711},
                {0.0528, 0.6039, 0.0000, -0.0406, 0.0406},
                {0.0305, 0.0000, 0.4852, -0.1421, 0.0812},
                {-0.0609, 0.1279, 0.0000, 0.4711, -0.0203},
                {0.2538, 0.0000, 0.0914, 0.0102, 0.5684}};
        double[] b = {0.7613, -0.8709, 3.2074, -1.8290, 2.9537};


        // Метод левой прогонки
        double[][] dA = {{0.6897,  -0.0908, 0.0000 , 0.0000,  0.0000},
                {0.0944,  1.0799 , 0.0000,  0.0000,  0.0000},
                {0.0000,  0.0000 , 0.8676,  -0.2541, 0.0000},
                {0.0000,  0.0000 , 0.0000,  0.8531,  -0.0363},
                {0.0000,  0.0000 , 0.0000,  0.0182,  1.0164}};
        double[] X = leftSweepMet(dA.clone(), b.clone());


        // Нахождение вектора невязки
        findResidua(dA.clone(), b.clone(), X);
    }


    // Метод левой прогонки
    public static double[] leftSweepMet(double[][] A, double[] b) {
        double[] diagonalC = new double[A.length];
        double[] diagonalA = new double[A.length];
        double[] diagonalB = new double[A.length];

        for (int i = 0; i < A.length; i++) {
            diagonalC[i] = A[i][i];
        }

        for (int i = 0; i < A.length - 1; i++) {
            diagonalA[i + 1] = -A[i + 1][i];
            diagonalB[i] = -A[i][i + 1];
        }

        System.out.println("\nДиагональ A:\n\t" + Arrays.toString(diagonalA));
        System.out.println("\nДиагональ B:\n\t" + Arrays.toString(diagonalB));
        System.out.println("\nДиагональ C:\n\t" + Arrays.toString(diagonalC));


        double[] alpha = new double[A.length];
        double[] beta = new double[A.length];

        alpha[A.length - 1] = diagonalA[A.length - 1] / diagonalC[A.length - 1];
        beta[A.length - 1] = b[A.length - 1] / diagonalC[A.length - 1];

        for (int i = A.length - 2; i >= 1 ; i--) {
            alpha[i] = diagonalA[i] / (diagonalC[i] - diagonalB[i] * alpha[i + 1]);
            beta[i] = (beta[i + 1] * diagonalB[i] + b[i]) / (diagonalC[i] - diagonalB[i] * alpha[i + 1]);
        }

        System.out.println("\nКоэффициенты Alpha:\n\t" + Arrays.toString(alpha));
        System.out.println("\nКоэффициенты Beta:\n\t" + Arrays.toString(beta));


        double[] massX = new double[A.length];
        massX[0] = (b[0] + beta[1] * diagonalB[0]) / (diagonalC[0] - alpha[1] * diagonalB[0]);

        for (int i = 1; i < A.length; i++) {
            massX[i] = alpha[i] * massX[i - 1] + beta[i];
        }

        System.out.println("\nВектор решений X:\n\t" + Arrays.toString(massX));

        return massX;
    }


    // Нахождение в-ра невязки (r = A*x - b)
    public static void findResidua(double[][] mtrA, double[] vecB, double[] massX) {
        double[] newVec = new double[vecB.length];
        double sum = 0;

        for (int i = 0; i < mtrA.length; i++) {
            for (int j = 0; j < mtrA.length; j++) {
                sum += mtrA[i][j] * massX[j];
            }
            newVec[i] = sum;
            sum = 0.0;
        }

        for (int i = 0; i < vecB.length; i++) {
            newVec[i] = newVec[i] - vecB[i];
        }

        System.out.println("\nВектор невязки:\n\t" + Arrays.toString(newVec));
    }
}