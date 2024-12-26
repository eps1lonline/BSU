#include <iostream>
#include <windows.h>

int n;
int minIndex, maxIndex; 
int min, max, average;

void min_max_thread(LPVOID* pointer)
{
	int* mass = (int*)pointer;
	min = mass[0];
	max = mass[0];

	for (int i = 0; i < n; i++) {
		if (mass[i] < min) {
			min = mass[i];
			minIndex = i;
		}
		if (mass[i] > max) {
			max = mass[i];
			maxIndex = i;
		}
		Sleep(7);
	}

	std::cout << "\nMinimal: " << min
			<< "\nMaximal: " << max
			<< "\nWork min_max_thread completed";
}

void average_thread(LPVOID* pointer) {
	average = 0;

	int* mass = (int*)pointer;
	for (int i = 0; i < n; i++) {
		average += mass[i];
		Sleep(12);
	}
	average /= n;

	std::cout << "\nAverage: " << average
			<< "\nWork average_thread completed";
}

void main_thread() {
	std::cout << "\nInput massive size: ";
	std::cin >> n;

	std::cout << "\nInput massive elements: ";
	int* mass = new int[n];
	for (int i = 0; i < n; i++)
		std::cin >> mass[i];

	HANDLE min_max;
	DWORD index_min_max;
	min_max = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)min_max_thread, (void*)mass, 0, &index_min_max);
	WaitForSingleObject(min_max, INFINITE);
	CloseHandle(min_max);

	HANDLE Average;
	DWORD index_Average;
	Average = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)average_thread, (void*)mass, 0, &index_Average);
	WaitForSingleObject(Average, INFINITE);
	CloseHandle(Average);
	
	mass[minIndex] = average;
	mass[maxIndex] = average;

	std::cout << "\nMassive: ";
	for (int i = 0; i < n; i++) 
		std::cout << mass[i] << " ";

	std::cout << std::endl << "Work main_thread completed";
}

int main() {
	try {
		HANDLE main;
		DWORD IDmain;
		main = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)main_thread, NULL, 0, &IDmain);
		WaitForSingleObject(main, INFINITE);
		CloseHandle(main);
	}
	catch (const char* msg) {
		std::cout << "Error: " << msg;
	}

	return 0;
}