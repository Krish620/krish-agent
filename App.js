import 'react-native-gesture-handler';
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Text, AppRegistry } from 'react-native';

import DashboardScreen from './screens/DashboardScreen';
import CallLogsScreen from './screens/CallLogsScreen';
import SettingsScreen from './screens/SettingsScreen';

const Tab = createBottomTabNavigator();

function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={{
          tabBarStyle: { backgroundColor: '#0f0f0f', borderTopColor: '#222' },
          tabBarActiveTintColor: '#7c6fea',
          tabBarInactiveTintColor: '#666',
          headerStyle: { backgroundColor: '#0f0f0f' },
          headerTintColor: '#fff',
        }}
      >
        <Tab.Screen
          name="Dashboard"
          component={DashboardScreen}
          options={{ tabBarIcon: ({ color }) => <Text style={{ fontSize: 20 }}>🏠</Text> }}
        />
        <Tab.Screen
          name="Call Logs"
          component={CallLogsScreen}
          options={{ tabBarIcon: ({ color }) => <Text style={{ fontSize: 20 }}>📋</Text> }}
        />
        <Tab.Screen
          name="Settings"
          component={SettingsScreen}
          options={{ tabBarIcon: ({ color }) => <Text style={{ fontSize: 20 }}>⚙️</Text> }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

export default App;