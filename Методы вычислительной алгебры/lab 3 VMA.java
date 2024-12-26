import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double[][] A = {{0.3857, -0.0508, 0.0102, 0.0203, 0.0711},
                {0.0528, 0.6039, 0.0000, -0.0406, 0.0406},
                {0.0305, 0.0000, 0.4852, -0.1421, 0.0812},
                {-0.0609, 0.1279, 0.0000, 0.4711, -0.0203},
                {0.2538, 0.0000, 0.0914, 0.0102, 0.5684}};
        double[] b = {0.7613, -0.8709, 3.2074, -1.8290, 2.9537};

        // Метод Гаусса-Зейделя
        seidelMet(A.clone(), b.clone());
    }


    // Метод Гаусса-Зейделя
    public static void seidelMet(double[][] A, double[] b) {
        int k = 0;
        double normD;
        boolean exit = false;
        double eps = Math.pow(10, -5);

        double[] oX = b.clone(); // Нулевое приближение
        double[] d = new double[oX.length];
        double[] nX = new double[A.length];
        double[] mApprox = new double[oX.length];

        while (!exit) {
            k++;

            for (int i = 0; i < A.length; i++) {
                nX[i] = b[i];

                for (int j = 0; j < A.length; j++) {
                    if (i != j) {
                        if (j < i) {
                            nX[i] += (-1) * A[i][j] * nX[j + 1 - 1];
                        }
                        else {
                            nX[i] += (-1) * A[i][j] * oX[j + 1 - 1];
                        }
                    }
                }

                nX[i] /= A[i][i];
            }


            double approx;
            for (int i = 0; i < oX.length; i++) {
                approx = Math.abs(nX[i] - oX[i]) / Math.abs(nX[i]);
                mApprox[i] = approx;
            }


            normD = 0;
            for (int j = 0; j < oX.length; j++) {
                d[j] = nX[j] - oX[j];
                normD += Math.pow(d[j], 2);
            }
            exit = (Math.sqrt(normD) <= eps) ? true : false;


            for (int i = 0; i < oX.length; i++) {
                oX[i] = nX[i];
                nX[i] = 0;
            }
        }


        System.out.println("k = " + k);
        System.out.println("X = " + Arrays.toString(oX));
        System.out.println("r = " + Arrays.toString(mApprox));
    }


    // Нахождение в-ра невязки (r = A*x - b)
    public static void findResidua(double[][] A, double[] b, double[] x) {
        double[] nVec = new double[b.length];
        double sum = 0;

        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A.length; j++) {
                sum += A[i][j] * x[j];
            }
            nVec[i] = sum;
            sum = 0.0;
        }

        for (int i = 0; i < b.length; i++) {
            nVec[i] = nVec[i] - b[i];
        }

        System.out.println("r = " + Arrays.toString(nVec));
    }
}
