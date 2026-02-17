import { useState, useMemo, useCallback, useEffect } from 'react'
import {
  Paper,
  SimpleGrid,
  Text,
  Group,
  Stack,
  Box,
  Select,
  SegmentedControl,
  Badge,
  ThemeIcon,
  Divider,
  Code,
  ScrollArea,
  ActionIcon,
  Switch,
  rem,
  Tabs,
  Loader,
  Center,
  RingProgress,
  ColorSwatch,
  Tooltip,
  Progress
} from '@mantine/core'
import {
  IconChartBar,
  IconChartPie,
  IconChartDots,
  IconTerminal2,
  IconRefresh,
  IconVolume,
  IconVolumeOff,
  IconMicrophone,
  IconMicrophoneOff,
  IconWaveSine,
  IconActivity,
  IconMoodSmile,
  IconMoodSad,
  IconQuestionMark,
  IconChartAreaLine,
  IconGridDots
} from '@tabler/icons-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  Legend,
  ReferenceLine
} from 'recharts'
import type { AudioAnalysisData, AudioSceneStat, ClipData } from '../types'

// Color schemes
const COLORS = {
  cue: {
    speech: '#4fc3f7',
    moan: '#f06292',
    groan: '#ff8a65',
    sigh: '#81c784',
    whimper: '#ba68c8',
    agitation: '#ff5252',
    silence: '#607d8b',
    ambient: '#90a4ae',
    unknown: '#757575'
  },
  valence: {
    positive: '#40c057',
    negative: '#ff6b6b',
    neutral: '#868e96',
    ambiguous: '#f59f00'
  },
  trend: {
    rising: '#ff6b6b',
    falling: '#4fc3f7',
    stable: '#868e96',
    unknown: '#495057'
  },
  chart: ['#7c4dff', '#448aff', '#00bcd4', '#4caf50', '#ff9800', '#f44336', '#e91e63', '#9c27b0']
}

// Quadrant definitions for acoustic space
const ACOUSTIC_QUADRANTS = {
  'rising-rising': { label: 'Building Intensity', color: '#ff6b6b', x: 0.75, y: 0.75 },
  'rising-falling': { label: 'Tension Release', color: '#f59f00', x: 0.75, y: 0.25 },
  'falling-rising': { label: 'Calming Intensity', color: '#7c4dff', x: 0.25, y: 0.75 },
  'falling-falling': { label: 'De-escalation', color: '#40c057', x: 0.25, y: 0.25 },
  'stable-stable': { label: 'Steady State', color: '#868e96', x: 0.5, y: 0.5 }
}

interface AudioExplorerProps {
  clips?: ClipData[]
  loading?: boolean
}

type ViewMode = 'classification' | 'acoustic' | 'quadrant' | 'fusion'

export function AudioExplorer({ clips, loading }: AudioExplorerProps) {
  const [audioData, setAudioData] = useState<AudioAnalysisData | null>(null)
  const [loadingAudio, setLoadingAudio] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('classification')
  
  // Chart state
  const [cueChartType, setCueChartType] = useState<string>('pie')
  const [showSpeechOnly, setShowSpeechOnly] = useState(false)
  const [trendMetric, setTrendMetric] = useState<string>('pitch')
  
  const [queryLog, setQueryLog] = useState<string[]>(['> Audio Explorer initialized'])

  // Fetch audio data on mount
  useEffect(() => {
    const fetchAudioData = async () => {
      setLoadingAudio(true)
      setError(null)
      try {
        const res = await fetch('/api/audio')
        if (!res.ok) {
          throw new Error(`Failed to fetch audio data: ${res.status}`)
        }
        const data = await res.json()
        
        // Check if data is available (empty dataset vs real data)
        if (data.summary?.dataAvailable === false || data.summary?.totalScenes === 0) {
          setError(data.summary?.message || 'Audio analysis not available. Run audio_analyzer.py to generate data.')
          return
        }
        
        setAudioData(data)
        logQuery(`Loaded ${data.summary.totalScenes} scenes (${data.summary.scenesWithAudio} with audio)`)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
        logQuery('Error loading audio data')
      } finally {
        setLoadingAudio(false)
      }
    }
    
    fetchAudioData()
  }, [])

  const logQuery = useCallback((msg: string) => {
    setQueryLog(prev => [...prev.slice(-30), `> ${msg}`])
  }, [])

  // Prepare cue distribution chart data
  const cueChartData = useMemo(() => {
    if (!audioData) return []
    
    let data = Object.entries(audioData.cueDistribution)
      .map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
        fill: COLORS.cue[name as keyof typeof COLORS.cue] || COLORS.cue.unknown
      }))
      .filter(d => d.value > 0)
      .sort((a, b) => b.value - a.value)
    
    if (showSpeechOnly) {
      data = data.filter(d => d.name.toLowerCase() !== 'speech' && d.name.toLowerCase() !== 'silence')
    }
    
    return data
  }, [audioData, showSpeechOnly])

  // Prepare valence hint chart data
  const valenceChartData = useMemo(() => {
    if (!audioData) return []
    
    return Object.entries(audioData.valenceHintDistribution)
      .map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
        fill: COLORS.valence[name as keyof typeof COLORS.valence] || COLORS.valence.neutral
      }))
      .filter(d => d.value > 0)
      .sort((a, b) => b.value - a.value)
  }, [audioData])

  // Prepare trend distribution chart data
  const trendChartData = useMemo(() => {
    if (!audioData) return []
    
    const dist = trendMetric === 'pitch' 
      ? audioData.pitchTrendDistribution 
      : audioData.energyTrendDistribution
    
    return Object.entries(dist)
      .map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
        fill: COLORS.trend[name as keyof typeof COLORS.trend] || COLORS.trend.unknown
      }))
      .filter(d => d.value > 0)
  }, [audioData, trendMetric])

  // Prepare quadrant scatter data (pitch trend × energy trend)
  const quadrantData = useMemo(() => {
    if (!audioData) return []
    
    const trendToValue = (trend: string | undefined) => {
      switch (trend) {
        case 'rising': return 0.8
        case 'falling': return 0.2
        case 'stable': return 0.5
        default: return 0.5
      }
    }
    
    return audioData.sceneStats
      .filter(s => s.audioPresent && s.acousticProfile)
      .map(s => {
        const pitchTrend = s.acousticProfile?.pitchTrend || 'unknown'
        const energyTrend = s.acousticProfile?.energyTrend || 'unknown'
        
        // Add some jitter to avoid overlapping points
        const jitter = () => (Math.random() - 0.5) * 0.15
        
        return {
          x: trendToValue(pitchTrend) + jitter(),
          y: trendToValue(energyTrend) + jitter(),
          name: s.sceneName,
          pitchTrend,
          energyTrend,
          cue: s.classification?.dominantCue || 'unknown',
          valence: s.classification?.valenceHint || 'neutral'
        }
      })
  }, [audioData])

  // Audio presence rate
  const audioPresenceRate = useMemo(() => {
    if (!audioData || audioData.summary.totalScenes === 0) return 0
    return (audioData.summary.scenesWithAudio / audioData.summary.totalScenes) * 100
  }, [audioData])

  // Speech presence rate
  const speechPresenceRate = useMemo(() => {
    if (!audioData || audioData.summary.scenesWithAudio === 0) return 0
    return (audioData.summary.scenesWithSpeech / audioData.summary.scenesWithAudio) * 100
  }, [audioData])

  // Render chart based on type
  const renderChart = (data: any[], chartType: string, height = 300) => {
    if (!data.length) {
      return (
        <Center h={height}>
          <Text c="dimmed">No data available</Text>
        </Center>
      )
    }

    if (chartType === 'pie') {
      return (
        <ResponsiveContainer width="100%" height={height}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={100}
              paddingAngle={2}
              dataKey="value"
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
            <RechartsTooltip
              contentStyle={{ backgroundColor: '#1a1b1e', border: '1px solid #2c2e33', borderRadius: 8, color: '#fff' }}
            />
          </PieChart>
        </ResponsiveContainer>
      )
    }

    // Default bar chart
    const isHorizontal = data.length > 5
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout={isHorizontal ? 'vertical' : 'horizontal'} margin={{ top: 20, right: 30, bottom: 40, left: isHorizontal ? 80 : 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2c2e33" />
          {isHorizontal ? (
            <>
              <XAxis type="number" stroke="#868e96" fontSize={12} />
              <YAxis type="category" dataKey="name" stroke="#868e96" fontSize={11} width={70} />
            </>
          ) : (
            <>
              <XAxis dataKey="name" stroke="#868e96" fontSize={11} />
              <YAxis stroke="#868e96" fontSize={12} />
            </>
          )}
          <RechartsTooltip
            contentStyle={{ backgroundColor: '#1a1b1e', border: '1px solid #2c2e33', borderRadius: 8, color: '#fff' }}
          />
          <Bar dataKey="value" radius={isHorizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    )
  }

  // Render quadrant scatter plot
  const renderQuadrantScatter = () => {
    if (!quadrantData.length) {
      return (
        <Center h={400}>
          <Stack align="center" gap="md">
            <IconChartDots size={48} stroke={1.5} color="#868e96" />
            <Text c="dimmed">No acoustic data available for quadrant visualization</Text>
          </Stack>
        </Center>
      )
    }

    return (
      <ResponsiveContainer width="100%" height={450}>
        <ScatterChart margin={{ top: 40, right: 40, bottom: 60, left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2c2e33" />
          
          {/* Quadrant labels */}
          <ReferenceLine x={0.5} stroke="#373A40" strokeDasharray="5 5" />
          <ReferenceLine y={0.5} stroke="#373A40" strokeDasharray="5 5" />
          
          <XAxis
            type="number"
            dataKey="x"
            name="Pitch Trend"
            stroke="#868e96"
            fontSize={12}
            domain={[0, 1]}
            tickCount={5}
            tickFormatter={(v) => v === 0 ? 'Falling' : v === 1 ? 'Rising' : v === 0.5 ? 'Stable' : ''}
            label={{ value: 'Pitch Trend', position: 'bottom', offset: 35, fill: '#868e96', fontSize: 12 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Energy Trend"
            stroke="#868e96"
            fontSize={12}
            domain={[0, 1]}
            tickCount={5}
            tickFormatter={(v) => v === 0 ? 'Falling' : v === 1 ? 'Rising' : v === 0.5 ? 'Stable' : ''}
            label={{ value: 'Energy Trend', angle: -90, position: 'insideLeft', fill: '#868e96', fontSize: 12, offset: -10 }}
          />
          <ZAxis range={[40, 120]} />
          <RechartsTooltip
            contentStyle={{ backgroundColor: '#1a1b1e', border: '1px solid #2c2e33', borderRadius: 8, color: '#fff' }}
            formatter={(value: number, name: string) => [
              name === 'x' ? (value > 0.6 ? 'Rising' : value < 0.4 ? 'Falling' : 'Stable') :
              name === 'y' ? (value > 0.6 ? 'Rising' : value < 0.4 ? 'Falling' : 'Stable') : value,
              name === 'x' ? 'Pitch' : name === 'y' ? 'Energy' : name
            ]}
            labelFormatter={(_, payload) => {
              const data = payload?.[0]?.payload
              return data ? `${data.name} (${data.cue})` : ''
            }}
          />
          
          {/* Color by dominant cue */}
          {Object.keys(COLORS.cue).map(cue => {
            const cueData = quadrantData.filter(d => d.cue === cue)
            if (cueData.length === 0) return null
            return (
              <Scatter
                key={cue}
                name={cue.charAt(0).toUpperCase() + cue.slice(1)}
                data={cueData}
                fill={COLORS.cue[cue as keyof typeof COLORS.cue]}
              />
            )
          })}
          
          <Legend />
        </ScatterChart>
      </ResponsiveContainer>
    )
  }

  // Loading state
  if (loading || loadingAudio) {
    return (
      <Center h={400}>
        <Stack align="center" gap="md">
          <Loader size="lg" color="cyan" type="dots" />
          <Text c="dimmed">Loading audio analysis data...</Text>
        </Stack>
      </Center>
    )
  }

  // Error state
  if (error) {
    return (
      <Paper p="xl" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
        <Stack align="center" gap="md">
          <IconVolumeOff size={48} stroke={1.5} color="#ff6b6b" />
          <Text c="red.4" fw={500}>{error}</Text>
          <Text c="dimmed" size="sm">
            Run: <Code>python scripts/audio_analyzer.py</Code>
          </Text>
        </Stack>
      </Paper>
    )
  }

  // No data state
  if (!audioData) {
    return (
      <Paper p="xl" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
        <Stack align="center" gap="md">
          <IconVolume size={48} stroke={1.5} color="#868e96" />
          <Text c="dimmed">No audio data available</Text>
        </Stack>
      </Paper>
    )
  }

  return (
    <Stack gap="lg">
      {/* Summary Stats */}
      <SimpleGrid cols={{ base: 2, sm: 4 }}>
        <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
          <Group justify="space-between">
            <Stack gap={2}>
              <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Total Scenes</Text>
              <Text size="xl" fw={700}>{audioData.summary.totalScenes.toLocaleString()}</Text>
            </Stack>
            <ThemeIcon variant="light" color="cyan" size="lg" radius="md">
              <IconChartBar size={20} />
            </ThemeIcon>
          </Group>
        </Paper>
        
        <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
          <Group justify="space-between">
            <Stack gap={2}>
              <Text size="xs" c="dimmed" tt="uppercase" fw={600}>With Audio</Text>
              <Text size="xl" fw={700}>{audioData.summary.scenesWithAudio}</Text>
              <Text size="xs" c="dimmed">{audioPresenceRate.toFixed(1)}%</Text>
            </Stack>
            <RingProgress
              size={50}
              thickness={5}
              roundCaps
              sections={[{ value: audioPresenceRate, color: 'cyan' }]}
            />
          </Group>
        </Paper>
        
        <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
          <Group justify="space-between">
            <Stack gap={2}>
              <Text size="xs" c="dimmed" tt="uppercase" fw={600}>With Speech</Text>
              <Text size="xl" fw={700}>{audioData.summary.scenesWithSpeech}</Text>
              <Text size="xs" c="dimmed">{speechPresenceRate.toFixed(1)}% of audio</Text>
            </Stack>
            <ThemeIcon variant="light" color="violet" size="lg" radius="md">
              <IconMicrophone size={20} />
            </ThemeIcon>
          </Group>
        </Paper>
        
        <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
          <Group justify="space-between">
            <Stack gap={2}>
              <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Avg Duration</Text>
              <Text size="xl" fw={700}>{audioData.summary.avgDurationSeconds.toFixed(1)}s</Text>
            </Stack>
            <ThemeIcon variant="light" color="teal" size="lg" radius="md">
              <IconWaveSine size={20} />
            </ThemeIcon>
          </Group>
        </Paper>
      </SimpleGrid>

      {/* Avg Ratios */}
      <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
        <Text size="sm" fw={600} mb="md">Audio Content Breakdown (Average)</Text>
        <SimpleGrid cols={{ base: 1, sm: 3 }}>
          <Stack gap={4}>
            <Group justify="space-between">
              <Text size="sm">Speech</Text>
              <Text size="sm" fw={600}>{(audioData.summary.avgSpeechRatio * 100).toFixed(1)}%</Text>
            </Group>
            <Progress value={audioData.summary.avgSpeechRatio * 100} color="cyan" size="sm" radius="xl" />
          </Stack>
          <Stack gap={4}>
            <Group justify="space-between">
              <Text size="sm">Non-Verbal</Text>
              <Text size="sm" fw={600}>{((1 - audioData.summary.avgSpeechRatio - audioData.summary.avgSilenceRatio) * 100).toFixed(1)}%</Text>
            </Group>
            <Progress value={(1 - audioData.summary.avgSpeechRatio - audioData.summary.avgSilenceRatio) * 100} color="pink" size="sm" radius="xl" />
          </Stack>
          <Stack gap={4}>
            <Group justify="space-between">
              <Text size="sm">Silence</Text>
              <Text size="sm" fw={600}>{(audioData.summary.avgSilenceRatio * 100).toFixed(1)}%</Text>
            </Group>
            <Progress value={audioData.summary.avgSilenceRatio * 100} color="gray" size="sm" radius="xl" />
          </Stack>
        </SimpleGrid>
      </Paper>

      {/* View Mode Tabs */}
      <Tabs value={viewMode} onChange={(v) => { setViewMode(v as ViewMode); logQuery(`Switched to ${v} view`) }}>
        <Tabs.List>
          <Tabs.Tab value="classification" leftSection={<IconMicrophone size={16} />}>
            Classification
          </Tabs.Tab>
          <Tabs.Tab value="acoustic" leftSection={<IconWaveSine size={16} />}>
            Acoustic Features
          </Tabs.Tab>
          <Tabs.Tab value="quadrant" leftSection={<IconGridDots size={16} />}>
            Quadrant View
          </Tabs.Tab>
        </Tabs.List>

        {/* Classification View */}
        <Tabs.Panel value="classification" pt="md">
          <SimpleGrid cols={{ base: 1, md: 2 }}>
            {/* Non-Verbal Cue Distribution */}
            <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
              <Group justify="space-between" mb="md">
                <Group gap="md">
                  <Text size="sm" fw={600}>Dominant Cue Distribution</Text>
                  <SegmentedControl
                    value={cueChartType}
                    onChange={(v) => setCueChartType(v)}
                    data={[
                      { value: 'pie', label: <IconChartPie size={14} /> },
                      { value: 'bar', label: <IconChartBar size={14} /> },
                    ]}
                    size="xs"
                    styles={{ root: { backgroundColor: '#25262b' } }}
                  />
                </Group>
                <Switch
                  checked={showSpeechOnly}
                  onChange={(e) => setShowSpeechOnly(e.currentTarget.checked)}
                  label="Non-verbal only"
                  size="xs"
                  color="pink"
                />
              </Group>
              {renderChart(cueChartData, cueChartType)}
            </Paper>

            {/* Valence Hint Distribution */}
            <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
              <Text size="sm" fw={600} mb="md">Audio Valence Hints</Text>
              {renderChart(valenceChartData, 'bar', 300)}
              <Divider my="md" color="dark.5" />
              <Group gap="xs" wrap="wrap">
                <Group gap={4}>
                  <ColorSwatch color={COLORS.valence.positive} size={14} />
                  <Text size="xs">Positive</Text>
                </Group>
                <Group gap={4}>
                  <ColorSwatch color={COLORS.valence.negative} size={14} />
                  <Text size="xs">Negative</Text>
                </Group>
                <Group gap={4}>
                  <ColorSwatch color={COLORS.valence.neutral} size={14} />
                  <Text size="xs">Neutral</Text>
                </Group>
                <Group gap={4}>
                  <ColorSwatch color={COLORS.valence.ambiguous} size={14} />
                  <Text size="xs">Ambiguous</Text>
                </Group>
              </Group>
            </Paper>
          </SimpleGrid>

          {/* Cue breakdown table */}
          <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }} mt="lg">
            <Text size="sm" fw={600} mb="md">Non-Verbal Cue Breakdown</Text>
            <SimpleGrid cols={{ base: 2, sm: 4 }}>
              {Object.entries(audioData.cueDistribution)
                .sort(([, a], [, b]) => b - a)
                .map(([cue, count]) => {
                  const total = Object.values(audioData.cueDistribution).reduce((a, b) => a + b, 0)
                  const pct = total > 0 ? (count / total) * 100 : 0
                  return (
                    <Group key={cue} justify="space-between">
                      <Group gap="xs">
                        <ColorSwatch color={COLORS.cue[cue as keyof typeof COLORS.cue] || COLORS.cue.unknown} size={12} />
                        <Text size="sm">{cue.charAt(0).toUpperCase() + cue.slice(1)}</Text>
                      </Group>
                      <Group gap={4}>
                        <Text size="sm" fw={600}>{count}</Text>
                        <Text size="xs" c="dimmed">({pct.toFixed(1)}%)</Text>
                      </Group>
                    </Group>
                  )
                })}
            </SimpleGrid>
          </Paper>
        </Tabs.Panel>

        {/* Acoustic Features View */}
        <Tabs.Panel value="acoustic" pt="md">
          <SimpleGrid cols={{ base: 1, md: 2 }}>
            {/* Pitch/Energy Trend Distribution */}
            <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
              <Group justify="space-between" mb="md">
                <Text size="sm" fw={600}>Trend Distribution</Text>
                <Select
                  value={trendMetric}
                  onChange={(v) => { setTrendMetric(v || 'pitch'); logQuery(`Trend metric: ${v}`) }}
                  data={[
                    { value: 'pitch', label: 'Pitch Trend' },
                    { value: 'energy', label: 'Energy Trend' },
                  ]}
                  w={150}
                  size="xs"
                  styles={{ input: { backgroundColor: '#25262b', borderColor: '#373A40' } }}
                />
              </Group>
              {renderChart(trendChartData, 'bar', 280)}
              <Divider my="md" color="dark.5" />
              <Group gap="xs">
                <Group gap={4}>
                  <ColorSwatch color={COLORS.trend.rising} size={14} />
                  <Text size="xs">Rising</Text>
                </Group>
                <Group gap={4}>
                  <ColorSwatch color={COLORS.trend.falling} size={14} />
                  <Text size="xs">Falling</Text>
                </Group>
                <Group gap={4}>
                  <ColorSwatch color={COLORS.trend.stable} size={14} />
                  <Text size="xs">Stable</Text>
                </Group>
              </Group>
            </Paper>

            {/* Speech Ratio Histogram */}
            <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
              <Text size="sm" fw={600} mb="md">Speech Ratio Distribution</Text>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={audioData.speechRatioHistogram} margin={{ top: 20, right: 30, bottom: 40, left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2c2e33" />
                  <XAxis dataKey="range" stroke="#868e96" fontSize={11} />
                  <YAxis stroke="#868e96" fontSize={12} />
                  <RechartsTooltip
                    contentStyle={{ backgroundColor: '#1a1b1e', border: '1px solid #2c2e33', borderRadius: 8, color: '#fff' }}
                  />
                  <Bar dataKey="count" fill="#4fc3f7" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </SimpleGrid>

          {/* Duration Histogram */}
          <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }} mt="lg">
            <Text size="sm" fw={600} mb="md">Audio Duration Distribution</Text>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={audioData.durationHistogram} margin={{ top: 20, right: 30, bottom: 40, left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2c2e33" />
                <XAxis dataKey="range" stroke="#868e96" fontSize={11} />
                <YAxis stroke="#868e96" fontSize={12} />
                <RechartsTooltip
                  contentStyle={{ backgroundColor: '#1a1b1e', border: '1px solid #2c2e33', borderRadius: 8, color: '#fff' }}
                />
                <Bar dataKey="count" fill="#7c4dff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Tabs.Panel>

        {/* Quadrant View */}
        <Tabs.Panel value="quadrant" pt="md">
          <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
            <Group justify="space-between" mb="md">
              <Stack gap={2}>
                <Text size="sm" fw={600}>Acoustic Quadrant Space</Text>
                <Text size="xs" c="dimmed">Pitch Trend × Energy Trend (colored by dominant cue)</Text>
              </Stack>
              <Badge variant="light" color="grape">{quadrantData.length} points</Badge>
            </Group>
            
            {renderQuadrantScatter()}
            
            {/* Quadrant Legend */}
            <Divider my="md" color="dark.5" />
            <SimpleGrid cols={{ base: 2, sm: 4 }}>
              <Stack gap={4} align="center">
                <Text size="xs" fw={600} c="red.4">Top Right</Text>
                <Text size="xs" c="dimmed">Building Intensity</Text>
                <Text size="xs" c="dimmed">↑ Pitch, ↑ Energy</Text>
              </Stack>
              <Stack gap={4} align="center">
                <Text size="xs" fw={600} c="yellow.4">Bottom Right</Text>
                <Text size="xs" c="dimmed">Tension Release</Text>
                <Text size="xs" c="dimmed">↑ Pitch, ↓ Energy</Text>
              </Stack>
              <Stack gap={4} align="center">
                <Text size="xs" fw={600} c="violet.4">Top Left</Text>
                <Text size="xs" c="dimmed">Calming Intensity</Text>
                <Text size="xs" c="dimmed">↓ Pitch, ↑ Energy</Text>
              </Stack>
              <Stack gap={4} align="center">
                <Text size="xs" fw={600} c="green.4">Bottom Left</Text>
                <Text size="xs" c="dimmed">De-escalation</Text>
                <Text size="xs" c="dimmed">↓ Pitch, ↓ Energy</Text>
              </Stack>
            </SimpleGrid>
          </Paper>
        </Tabs.Panel>
      </Tabs>

      {/* Query Log */}
      <Paper p="md" radius="md" bg="dark.9" withBorder style={{ borderColor: '#2c2e33' }}>
        <Group justify="space-between" mb="sm">
          <Group gap="xs">
            <ThemeIcon variant="light" color="green" size="sm">
              <IconTerminal2 size={14} />
            </ThemeIcon>
            <Text size="sm" fw={600} c="green.4" style={{ fontFamily: 'monospace' }}>
              Query Log
            </Text>
          </Group>
          <ActionIcon variant="subtle" color="gray" onClick={() => setQueryLog(['> Cleared'])}>
            <IconRefresh size={14} />
          </ActionIcon>
        </Group>
        <ScrollArea h={60}>
          <Stack gap={2}>
            {queryLog.slice(-8).map((log, i) => (
              <Code key={i} block style={{
                backgroundColor: 'transparent',
                color: log.includes('Error') ? '#ff6b6b' : '#8ce99a',
                fontSize: 11,
                padding: '2px 0'
              }}>
                {log}
              </Code>
            ))}
          </Stack>
        </ScrollArea>
      </Paper>
    </Stack>
  )
}
