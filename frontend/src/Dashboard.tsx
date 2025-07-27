import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface Performance {
  total_profit: number
  win_rate: number
  total_trades: number
  winning_trades: number
  losing_trades: number
}

interface Position {
  pool: string
  apr: number
  value_usd: number
  pending_rewards?: number
}

interface RecentAction {
  timestamp: string
  type: string
  description: string
  profit?: number
}

interface AgentStatus {
  status: 'learning' | 'trading' | 'rebalancing'
  observation_days_left?: number
  emotions: {
    confidence: number
    curiosity: number
    caution: number
    satisfaction: number
  }
}

const Dashboard: React.FC = () => {
  const [performance, setPerformance] = useState<Performance | null>(null)
  const [positions, setPositions] = useState<Position[]>([])
  const [recentActions, setRecentActions] = useState<RecentAction[]>([])
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null)
  const [totalValue, setTotalValue] = useState(0)
  const [gasPrice, setGasPrice] = useState(0)
  const [connected, setConnected] = useState(false)

  // Fetch data from API
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Get performance
        const perfResponse = await axios.get('/api/performance/24h')
        setPerformance(perfResponse.data)

        // Get positions
        const posResponse = await axios.get('/api/positions')
        setPositions(posResponse.data.positions || [])
        setTotalValue(posResponse.data.total_value || 0)

        // Get health/status
        const healthResponse = await axios.get('/api/health')
        setConnected(healthResponse.data.agent_active)
      } catch (error) {
        console.error('Error fetching data:', error)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [])

  // WebSocket for real-time updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/live')
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'status') {
        setAgentStatus({
          status: 'trading', // Would come from actual data
          emotions: data.emotions
        })
        setGasPrice(data.gas || 0)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    return () => ws.close()
  }, [])

  // Mock recent actions for demo
  useEffect(() => {
    setRecentActions([
      {
        timestamp: new Date().toISOString(),
        type: 'rebalance',
        description: 'Rebalanced USDC/DAI → WETH/USDC',
        profit: 45
      },
      {
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        type: 'compound',
        description: 'Compounded 125 AERO rewards'
      },
      {
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        type: 'arbitrage',
        description: 'Arbitrage opportunity captured',
        profit: 32
      }
    ])
  }, [])

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    return `${Math.floor(hours / 24)}d ago`
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'learning': return 'text-yellow-400'
      case 'trading': return 'text-green-400'
      case 'rebalancing': return 'text-blue-400'
      default: return 'text-gray-400'
    }
  }

  const getActionIcon = (type: string) => {
    switch (type) {
      case 'rebalance': return '🔄'
      case 'compound': return '💎'
      case 'arbitrage': return '⚡'
      default: return '📊'
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Athena AI Dashboard</h1>
          <div className="flex items-center gap-4">
            <span className={`text-lg ${connected ? 'text-green-400' : 'text-red-400'}`}>
              {connected ? '● Connected' : '● Disconnected'}
            </span>
            {agentStatus && (
              <span className={`text-lg ${getStatusColor(agentStatus.status)}`}>
                Status: {agentStatus.status.charAt(0).toUpperCase() + agentStatus.status.slice(1)}
                {agentStatus.observation_days_left && ` (${agentStatus.observation_days_left}d left)`}
              </span>
            )}
            <span className="text-gray-400">Gas: {gasPrice.toFixed(0)} gwei</span>
          </div>
        </div>

        {/* Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-gray-400 text-sm mb-2">Total Value</h3>
            <p className="text-3xl font-bold text-white">${totalValue.toLocaleString()}</p>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-gray-400 text-sm mb-2">24h Profit</h3>
            <p className="text-3xl font-bold text-green-400">
              +${performance?.total_profit.toFixed(2) || '0'}
            </p>
            <p className="text-sm text-gray-400">
              {performance && performance.total_profit > 0 
                ? `+${((performance.total_profit / totalValue) * 100).toFixed(1)}%`
                : '0%'
              }
            </p>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-gray-400 text-sm mb-2">Win Rate</h3>
            <p className="text-3xl font-bold text-white">
              {performance ? `${(performance.win_rate * 100).toFixed(0)}%` : '0%'}
            </p>
            <p className="text-sm text-gray-400">
              {performance ? `${performance.winning_trades}/${performance.total_trades} trades` : '0/0 trades'}
            </p>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-gray-400 text-sm mb-2">Active Positions</h3>
            <p className="text-3xl font-bold text-white">{positions.length}</p>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Positions */}
          <div className="lg:col-span-2 bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold text-white mb-4">Active Positions</h2>
            <div className="space-y-4">
              {positions.length > 0 ? positions.map((position, i) => (
                <div key={i} className="bg-gray-700 rounded-lg p-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{position.pool}</h3>
                      <p className="text-gray-400">Value: ${position.value_usd.toLocaleString()}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-2xl font-bold text-green-400">{position.apr.toFixed(1)}%</p>
                      <p className="text-sm text-gray-400">APR</p>
                    </div>
                  </div>
                  {position.pending_rewards && position.pending_rewards > 0 && (
                    <div className="mt-2 text-sm text-yellow-400">
                      Pending rewards: {position.pending_rewards.toFixed(2)} AERO
                    </div>
                  )}
                </div>
              )) : (
                <p className="text-gray-400">No active positions</p>
              )}
            </div>
          </div>

          {/* Recent Actions */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold text-white mb-4">Recent Actions</h2>
            <div className="space-y-3">
              {recentActions.map((action, i) => (
                <div key={i} className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">{getActionIcon(action.type)}</span>
                    <div className="flex-1">
                      <p className="text-white text-sm">{action.description}</p>
                      {action.profit && (
                        <p className="text-green-400 text-sm">+${action.profit.toFixed(2)} profit</p>
                      )}
                      <p className="text-gray-500 text-xs">{formatTime(action.timestamp)}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Emotional State */}
        {agentStatus && (
          <div className="mt-6 bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold text-white mb-4">Athena's State of Mind</h2>
            <div className="grid grid-cols-4 gap-4">
              {Object.entries(agentStatus.emotions).map(([emotion, value]) => (
                <div key={emotion} className="text-center">
                  <p className="text-gray-400 capitalize mb-2">{emotion}</p>
                  <div className="relative h-32 bg-gray-700 rounded-lg overflow-hidden">
                    <div 
                      className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-athena-primary to-athena-secondary transition-all duration-500"
                      style={{ height: `${value * 100}%` }}
                    />
                    <span className="absolute inset-0 flex items-center justify-center text-white font-bold">
                      {(value * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard