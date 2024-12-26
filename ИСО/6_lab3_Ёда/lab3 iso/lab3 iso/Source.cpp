#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

ifstream fin("input.txt");
ofstream fout("output.txt");

int n;
int border;

vector<int> mt;
vector<bool> used;
vector<bool> block;
vector<vector<int>> vec;

int result[256];
int e[256][256];
int parent[256];
int mtr[256][256];



bool try_kuhn(int v) {
    if (used[v])
    {
        return false;
    }

    used[v] = true;

    for (int i = 0; i < vec[v].size(); i++) 
    {
        int to = vec[v][i];

        if (mt[to] == -1 || try_kuhn(mt[to])) 
        {
            mt[to] = v;
            return true;
        }
    }

    return false;
}

bool function(int m) 
{
    vec.clear();

    for (int i = 0; i < n; i++) 
    {
        vector<int> cur;

        for (int j = 0; j < n; j++) 
        {
            if (mtr[i][j] >= m) cur.push_back(j);
        }
        vec.push_back(cur);
    }

    mt.assign(n, -1);

    for (int v = 0; v < n; ++v) 
    {
        used.assign(n, false);
        try_kuhn(v);
    }

    int ans = 0;

    for (int i = 0; i < n; ++i)
    {
        if (mt[i] != -1) ans++;
    }

    if (ans == n) 
    {
        return true;
    }
    else 
    {
        return false;
    }
}

void dfs(int v, int p) 
{
    if (used[v])
    {
        return;
    }

    used[v] = true;
    parent[v] = p;

    for (int i = 0; i < 2 * n; i++) 
    {
        if (e[v][i] == 1 && !block[i]) dfs(i, v);
    }
}

void findLex(int v) 
{
    for (int i = 0; i < n; i++)
    {
        e[i][result[i] + n] = 1;
    }

    used.assign(2 * n, false);

    for (int i = 0; i < n; i++)
    {
        parent[i] = -1;
    }

    dfs(v, v);

    for (int i = 0; i < n; i++)
    {
        e[i][result[i] + n] = 0;
    }

    for (int i = n; i < 2 * n; i++) 
    {
        if (used[i] && !block[i] && mtr[v][i - n] >= border) 
        {
            block[i] = true;    
            result[v] = i - n;

            int u = parent[i];

            while (u != parent[u]) 
            {
                result[u] = parent[u] - n;
                u = parent[parent[u]];
            }
            break;
        }
    }
    block[v] = true;
}

void lex() 
{
    block.assign(2 * n, false);

    for (int i = 0; i < n; i++)
    {
        result[mt[i]] = i;
    }

    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            if (mtr[i][j] >= border) e[n + j][i] = 1;
        }
    }

    for (int i = 0; i < n; i++) 
    {
        findLex(i);
    }

    for (int i = 0; i < n - 1; i++)
    {
        fout << result[i] + 1 << " ";
    }

    fout << result[n - 1] + 1 << endl;
}

int main() 
{
    fin >> n;

    int left = 1;
    int right = 0;

    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            fin >> mtr[i][j];

            if (mtr[i][j] > right)
            {
                right = mtr[i][j];
            }
        }
    }

    while (right - left > 1) 
    {
        int mid = (left + right) / 2;

        if (function(mid)) 
        {
            left = mid;
        }
        else 
        {
            right = mid;
        }
    }

    if (function(right)) 
    {
        border = right;
    }
    else 
    {
        function(left);
        border = left;
    }

    fout << border << endl;

    lex();
}
