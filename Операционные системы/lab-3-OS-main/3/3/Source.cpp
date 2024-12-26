#include <mutex>
#include <vector>
#include <iostream>
#include <Windows.h>

using namespace std;

volatile int* mass;
volatile int massCount;
volatile bool* thrTerm;

volatile HANDLE coutMutex;
volatile HANDLE* startEvents;
volatile HANDLE* contEvents;
volatile HANDLE* termEvents;
volatile HANDLE* workStopEvents;

DWORD WINAPI marker(LPVOID wrappedNum) {
	int n;
	memcpy(&n, &wrappedNum, sizeof n);

	WaitForSingleObject(startEvents[n], INFINITE);
	ResetEvent(startEvents[n]);

	srand(n);
	while (true) {
		int i = rand();
		i %= massCount;

		if (mass[i] == 0) {
			Sleep(5);
			mass[i] = n + 1;
			Sleep(5);
		}
		else {
			vector<int> markedIndices;

			for (int i = 0; i < massCount; i++) 
				if (mass[i] == n) 
					markedIndices.push_back(i);

			WaitForSingleObject(coutMutex, INFINITE);
			cout << "Thread " << n + 1 << ": marked " << markedIndices.size() << ", forbidden index " << i << '\n';
			ReleaseMutex(coutMutex);

			SetEvent(workStopEvents[n]);
			HANDLE* waitingHandles = new HANDLE[2];
			waitingHandles[0] = contEvents[n];
			waitingHandles[1] = termEvents[n];
			DWORD response = WaitForMultipleObjects(2, waitingHandles, false, INFINITE);

			ResetEvent(contEvents[n]);
			ResetEvent(termEvents[n]);

			if (response == 1) {
				for (int i = 0; i < markedIndices.size(); i++) 
					mass[markedIndices[i]] = 0;

				return 0;
			}
		}
	}
}

int main() {
	coutMutex = CreateMutex(NULL, FALSE, NULL);

	int n;
	cout << "Enter number of elements: ";
	cin >> n;

	int* array = new int[n];
	cout << "Enter " << n << " elements: ";
	for (int i = 0; i < n; i++) 
		cin >> array[i];

	massCount = n;
	mass = array;

	int thrCount;
	cout << "Enter number of threads: ";
	cin >> thrCount;

	thrTerm = new bool[thrCount];
	for (int i = 0; i < thrCount; i++)
		thrTerm[i] = false;

	startEvents = new HANDLE[thrCount];
	workStopEvents = new HANDLE[thrCount];
	termEvents = new HANDLE[thrCount];
	contEvents = new HANDLE[thrCount];

	for (int i = 0; i < thrCount; i++) {
		startEvents[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
		workStopEvents[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
		termEvents[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
		contEvents[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
	}

	HANDLE* threadHandles = new HANDLE[thrCount];

	for (int i = 0; i < thrCount; i++) {
		threadHandles[i] = CreateThread(NULL, 0, marker, (LPVOID)i, 0, NULL);
	}

	for (int i = 0; i < thrCount; i++) {
		SetEvent(startEvents[i]);
	}

	for (int z = 0; z < thrCount; z++) {
		for (int i = 0; i < thrCount; i++) {
			if (thrTerm[i])
				continue;

			HANDLE* waitingEvents = new HANDLE[2];
			waitingEvents[0] = workStopEvents[i];
			waitingEvents[1] = termEvents[i];
			WaitForMultipleObjects(2, waitingEvents, false, INFINITE);
		}
		for (int i = 0; i < thrCount; i++) {
			ResetEvent(workStopEvents[i]);
		}

		cout << "Modified array:\n";
		for (int i = 0; i < n; i++) {
			cout << array[i] << " ";
		}
		cout << '\n';

		int terminateMarker = -1;
		while (true) {
			cout << "Enter ID of thread to terminate: ";
			cin >> terminateMarker;
			terminateMarker--;

			if (terminateMarker >= 0 && terminateMarker < thrCount && !thrTerm[terminateMarker]) {
				break;
			}
			cout << "Unknown thread ID\n";
		}

		thrTerm[terminateMarker] = true;
		SetEvent(termEvents[terminateMarker]);
		WaitForSingleObject(threadHandles[terminateMarker], INFINITE);

		cout << "Thread " << terminateMarker + 1 << " terminated\n";
		cout << "Modified array:\n";
		for (int i = 0; i < n; i++) {
			cout << array[i] << " ";
		}
		cout << '\n';

		for (int i = 0; i < thrCount; i++) {
			SetEvent(contEvents[i]);
		}
	}

	return 0;
}