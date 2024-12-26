import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double[][] A = {{0.3857, -0.0508, 0.0102, 0.0203, 0.0711},
                {0.0528, 0.6039, 0.0000, -0.0406, 0.0406},
                {0.0305, 0.0000, 0.4852, -0.1421, 0.0812},
                {-0.0609, 0.1279, 0.0000, 0.4711, -0.0203},
                {0.2538, 0.0000, 0.0914, 0.0102, 0.5684}};
        double[] b = {0.7613, -0.8709, 3.2074, -1.8290, 2.9537};
        double[] oldX = b.clone();

        lowerRelax(A.clone(), b.clone(), oldX, 0.25);
        lowerRelax(A.clone(), b.clone(), oldX, 0.5);
        lowerRelax(A.clone(), b.clone(), oldX, 0.75);
    }


    // Метод нижней релаксации
    public static void lowerRelax(double[][] A, double[] b, double[] oX, double w) {
        int k = 0;
        double normD;
        boolean exit = false;
        double eps = Math.pow(10, -5);
        double[] nX = new double[A.length];
        double[] d = new double[oX.length];

        while (!exit) {
            k++;

            for (int i = 0; i < A.length; i++) {
                double s1 = 0, s2 = 0;

                for (int j = 0; j < i; j++) {
                    s1 += (A[i][j] * nX[j]) / A[i][i];
                }
                for (int j = i + 1; j < A.length; j++) {
                    s2 += (A[i][j] * oX[j]) / A[i][i];
                }

                nX[i] = (1 - w) * oX[i] - w * s1 - w * s2 + (w * b[i]) / A[i][i];
            }


            normD = 0;
            for (int j = 0; j < oX.length; j++) {
                d[j] = nX[j] - oX[j];
                normD += Math.pow(d[j], 2);
            }
            if (Math.sqrt(normD) <= eps) {
                exit = true;
            }
            oX = nX.clone();
        }

        System.out.println("ω = " + w);
        System.out.println("k = " + k);
        System.out.println("X = " + Arrays.toString(nX));
        findResidua(A.clone(), b.clone(), nX);
        System.out.println();
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
