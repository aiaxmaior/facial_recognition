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
  Tooltip,
  Switch,
  rem,
  Collapse,
  Button,
  Tabs,
  Loader,
  Center,
  RingProgress,
  ColorSwatch
} from '@mantine/core'
import {
  IconChartBar,
  IconChartPie,
  IconChartDots,
  IconChartHistogram,
  IconChartAreaLine,
  IconAdjustments,
  IconTerminal2,
  IconRefresh,
  IconUsers,
  IconUserScan,
  IconActivity,
  IconBabyCarriage,
  IconMan,
  IconWoman,
  IconChevronDown,
  IconChevronUp,
  IconEye,
  IconEyeOff,
  IconArrowsExchange,
  IconFilter,
  IconTrendingUp,
  IconBodyScan,
  IconMoodSmile
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
  AreaChart,
  Area,
  ReferenceLine,
  Brush
} from 'recharts'
import type { 
  DemographicsData, 
  NudenetData, 
  ExplorerData, 
  UnifiedSceneData,
  FeatureDefinition,
  ClipData 
} from '../types'

// Rich color palette
const COLORS = {
  age: {
    infant: '#ff6b9d',
    child: '#c44dff',
    adolescent: '#7c4dff',
    young_adult: '#448aff',
    adult: '#00bcd4',
    middle_aged: '#4caf50',
    senior: '#ff9800',
    unknown: '#607d8b'
  },
  gender: {
    Male: '#4fc3f7',
    Female: '#f06292',
    unknown: '#78909c'
  },
  nudenet: {
    exposed: '#ff5252',
    covered: '#69f0ae',
    detection: '#ffab40'
  },
  emotion: {
    happy: '#40c057',
    sad: '#748ffc',
    angry: '#ff6b6b',
    fear: '#f59f00',
    surprise: '#da77f2',
    disgust: '#20c997',
    neutral: '#868e96',
  },
  chart: ['#7c4dff', '#448aff', '#00bcd4', '#4caf50', '#ff9800', '#f44336', '#e91e63', '#9c27b0', '#3f51b5', '#009688']
}

// Nudenet label display names
const NUDENET_LABEL_NAMES: Record<string, string> = {
  'FEMALE_GENITALIA_EXPOSED': 'Female Genitalia (Exp)',
  'MALE_GENITALIA_EXPOSED': 'Male Genitalia (Exp)',
  'BUTTOCKS_EXPOSED': 'Buttocks (Exp)',
  'FEMALE_BREAST_EXPOSED': 'Female Breast (Exp)',
  'ANUS_EXPOSED': 'Anus (Exp)',
  'BELLY_EXPOSED': 'Belly (Exp)',
  'ARMPITS_EXPOSED': 'Armpits (Exp)',
  'FEET_EXPOSED': 'Feet (Exp)',
  'FEMALE_GENITALIA_COVERED': 'Female Genitalia (Cov)',
  'BUTTOCKS_COVERED': 'Buttocks (Cov)',
  'FEMALE_BREAST_COVERED': 'Female Breast (Cov)',
  'ANUS_COVERED': 'Anus (Cov)',
  'BELLY_COVERED': 'Belly (Cov)',
  'ARMPITS_COVERED': 'Armpits (Cov)',
  'FEET_COVERED': 'Feet (Cov)',
}

interface DataExplorerProps {
  demographics: DemographicsData | null
  clips: ClipData[]
  loading?: boolean
}

type ViewMode = 'demographics' | 'nudenet' | 'compare'

export function DataExplorer({ demographics, clips, loading }: DataExplorerProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('demographics')
  const [nudenetData, setNudenetData] = useState<NudenetData | null>(null)
  const [explorerData, setExplorerData] = useState<ExplorerData | null>(null)
  const [loadingExtra, setLoadingExtra] = useState(false)
  
  // Comparison state
  const [xFeature, setXFeature] = useState<string>('meanAge')
  const [yFeature, setYFeature] = useState<string>('nudenetDetectionRate')
  const [colorByFeature, setColorByFeature] = useState<string>('dominantEmotion')
  const [showTrendline, setShowTrendline] = useState(false)
  const [filterPersonPresent, setFilterPersonPresent] = useState(false)
  
  // Demographics chart state
  const [demoMetric, setDemoMetric] = useState<string>('ageDistribution')
  const [demoChartType, setDemoChartType] = useState<string>('bar')
  
  // Nudenet chart state
  const [nudenetMetric, setNudenetMetric] = useState<string>('labelDistribution')
  const [showExposedOnly, setShowExposedOnly] = useState(false)
  
  const [queryLog, setQueryLog] = useState<string[]>(['> Data Explorer initialized'])
  
  // Fetch additional data on mount
  useEffect(() => {
    const fetchExtraData = async () => {
      setLoadingExtra(true)
      try {
        const [nudenetRes, explorerRes] = await Promise.all([
          fetch('/api/nudenet').catch(() => null),
          fetch('/api/explorer-data').catch(() => null)
        ])
        
        if (nudenetRes?.ok) {
          const data = await nudenetRes.json()
          setNudenetData(data)
          logQuery('Loaded NudeNet data')
        }
        
        if (explorerRes?.ok) {
          const data = await explorerRes.json()
          setExplorerData(data)
          logQuery(`Loaded ${data.totalScenes} scenes for comparison`)
        }
      } catch (err) {
        logQuery('Error loading additional data')
      } finally {
        setLoadingExtra(false)
      }
    }
    
    fetchExtraData()
  }, [])
  
  const logQuery = useCallback((msg: string) => {
    setQueryLog(prev => [...prev.slice(-30), `> ${msg}`])
  }, [])

  // Get feature options for dropdowns
  const featureOptions = useMemo(() => {
    if (!explorerData) return []
    return explorerData.numericFeatures.map(f => ({
      value: f.key,
      label: f.label,
      group: f.category.charAt(0).toUpperCase() + f.category.slice(1)
    }))
  }, [explorerData])

  const colorByOptions = useMemo(() => {
    const options = [
      { value: 'none', label: 'None', group: 'General' },
      { value: 'dominantEmotion', label: 'Dominant Emotion', group: 'Emotions' },
      { value: 'personPresent', label: 'Person Present', group: 'Detection' },
    ]
    return options
  }, [])

  // Prepare scatter plot data for comparison
  const comparisonData = useMemo(() => {
    if (!explorerData?.scenes) return []
    
    let filtered = explorerData.scenes.filter(scene => {
      const xVal = scene[xFeature as keyof UnifiedSceneData]
      const yVal = scene[yFeature as keyof UnifiedSceneData]
      if (xVal === null || xVal === undefined || yVal === null || yVal === undefined) return false
      if (filterPersonPresent && !scene.personPresent) return false
      return true
    })
    
    return filtered.map(scene => {
      const xVal = scene[xFeature as keyof UnifiedSceneData] as number
      const yVal = scene[yFeature as keyof UnifiedSceneData] as number
      let colorVal = colorByFeature === 'none' ? 'all' : scene[colorByFeature as keyof UnifiedSceneData]
      
      return {
        x: xVal,
        y: yVal,
        colorBy: String(colorVal || 'unknown'),
        name: scene.sceneName,
        emotion: scene.dominantEmotion
      }
    })
  }, [explorerData, xFeature, yFeature, colorByFeature, filterPersonPresent])

  // Calculate correlation coefficient
  const correlation = useMemo(() => {
    if (comparisonData.length < 3) return null
    
    const n = comparisonData.length
    const sumX = comparisonData.reduce((a, b) => a + b.x, 0)
    const sumY = comparisonData.reduce((a, b) => a + b.y, 0)
    const sumXY = comparisonData.reduce((a, b) => a + b.x * b.y, 0)
    const sumX2 = comparisonData.reduce((a, b) => a + b.x * b.x, 0)
    const sumY2 = comparisonData.reduce((a, b) => a + b.y * b.y, 0)
    
    const num = n * sumXY - sumX * sumY
    const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))
    
    return den === 0 ? 0 : num / den
  }, [comparisonData])

  // Demographics chart data
  const demographicsChartData = useMemo(() => {
    if (!demographics) return []
    
    switch (demoMetric) {
      case 'ageDistribution':
        return Object.entries(demographics.ageDistribution)
          .map(([name, value]) => ({
            name: name.replace('_', ' '),
            value,
            fill: COLORS.age[name as keyof typeof COLORS.age] || COLORS.chart[0]
          }))
          .filter(d => d.value > 0)
          .sort((a, b) => b.value - a.value)
      
      case 'genderDistribution':
        return Object.entries(demographics.genderDistribution)
          .map(([name, value]) => ({
            name,
            value,
            fill: COLORS.gender[name as keyof typeof COLORS.gender] || COLORS.chart[0]
          }))
          .filter(d => d.value > 0)
      
      case 'raceDistribution':
        return Object.entries(demographics.raceDistribution)
          .map(([name, value]) => ({
            name: name.replace('_', ' '),
            value,
            fill: COLORS.chart[Object.keys(demographics.raceDistribution).indexOf(name) % COLORS.chart.length]
          }))
          .filter(d => d.value > 0)
          .sort((a, b) => b.value - a.value)
      
      case 'ageHistogram':
        return demographics.ageHistogram.map((bin, idx) => ({
          name: bin.range,
          value: bin.count,
          fill: COLORS.chart[idx % COLORS.chart.length]
        }))
      
      default:
        return []
    }
  }, [demographics, demoMetric])

  // Nudenet chart data
  const nudenetChartData = useMemo(() => {
    if (!nudenetData) return []
    
    switch (nudenetMetric) {
      case 'labelDistribution':
        const labels = showExposedOnly 
          ? nudenetData.exposedDistribution 
          : nudenetData.labelDistribution
        return Object.entries(labels)
          .map(([name, value]) => ({
            name: NUDENET_LABEL_NAMES[name] || name,
            fullName: name,
            value,
            fill: name.includes('EXPOSED') ? COLORS.nudenet.exposed : COLORS.nudenet.covered
          }))
          .sort((a, b) => b.value - a.value)
          .slice(0, 15)
      
      case 'exposedVsCovered':
        const exposed = Object.values(nudenetData.exposedDistribution).reduce((a, b) => a + b, 0)
        const covered = Object.values(nudenetData.coveredDistribution).reduce((a, b) => a + b, 0)
        return [
          { name: 'Exposed', value: exposed, fill: COLORS.nudenet.exposed },
          { name: 'Covered', value: covered, fill: COLORS.nudenet.covered }
        ]
      
      case 'detectionRateHistogram':
        return nudenetData.detectionRateHistogram.map((bin, idx) => ({
          name: bin.range,
          value: bin.count,
          fill: COLORS.chart[idx % COLORS.chart.length]
        }))
      
      default:
        return []
    }
  }, [nudenetData, nudenetMetric, showExposedOnly])

  // Get unique color values for legend
  const colorCategories = useMemo(() => {
    if (colorByFeature === 'none') return []
    const unique = [...new Set(comparisonData.map(d => d.colorBy))]
    return unique.filter(v => v !== 'unknown').sort()
  }, [comparisonData, colorByFeature])

  const getColorForCategory = (category: string) => {
    if (colorByFeature === 'dominantEmotion') {
      return COLORS.emotion[category as keyof typeof COLORS.emotion] || '#868e96'
    }
    if (colorByFeature === 'personPresent') {
      return category === 'true' ? '#4caf50' : '#ff5252'
    }
    return '#7c4dff'
  }

  // Render chart based on type
  const renderChart = (data: any[], chartType: string, height = 350) => {
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
              innerRadius={60}
              outerRadius={120}
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
    const isHorizontal = data.length > 6
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout={isHorizontal ? 'vertical' : 'horizontal'} margin={{ top: 20, right: 30, bottom: 60, left: isHorizontal ? 120 : 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2c2e33" />
          {isHorizontal ? (
            <>
              <XAxis type="number" stroke="#868e96" fontSize={12} />
              <YAxis type="category" dataKey="name" stroke="#868e96" fontSize={11} width={110} />
            </>
          ) : (
            <>
              <XAxis dataKey="name" stroke="#868e96" fontSize={11} angle={-45} textAnchor="end" height={80} />
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

  // Render comparison scatter plot
  const renderComparisonScatter = () => {
    if (!comparisonData.length) {
      return (
        <Center h={400}>
          <Stack align="center" gap="md">
            <IconChartDots size={48} stroke={1.5} color="#868e96" />
            <Text c="dimmed">No data available for comparison</Text>
            <Text c="dimmed" size="sm">Select different features or adjust filters</Text>
          </Stack>
        </Center>
      )
    }

    const xLabel = featureOptions.find(f => f.value === xFeature)?.label || xFeature
    const yLabel = featureOptions.find(f => f.value === yFeature)?.label || yFeature

    return (
      <ResponsiveContainer width="100%" height={450}>
        <ScatterChart margin={{ top: 20, right: 30, bottom: 60, left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2c2e33" />
          <XAxis 
            type="number" 
            dataKey="x" 
            name={xLabel}
            stroke="#868e96"
            fontSize={12}
            label={{ value: xLabel, position: 'bottom', offset: 40, fill: '#868e96', fontSize: 12 }}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name={yLabel}
            stroke="#868e96"
            fontSize={12}
            label={{ value: yLabel, angle: -90, position: 'insideLeft', fill: '#868e96', fontSize: 12, offset: -10 }}
          />
          <ZAxis range={[40, 160]} />
          <RechartsTooltip
            contentStyle={{ backgroundColor: '#1a1b1e', border: '1px solid #2c2e33', borderRadius: 8, color: '#fff' }}
            formatter={(value: number, name: string) => [
              typeof value === 'number' ? value.toFixed(3) : value,
              name === 'x' ? xLabel : name === 'y' ? yLabel : name
            ]}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.name || ''}
          />
          {colorByFeature === 'none' ? (
            <Scatter name="Scenes" data={comparisonData} fill="#7c4dff" />
          ) : (
            colorCategories.map((category, idx) => (
              <Scatter
                key={category}
                name={category}
                data={comparisonData.filter(d => d.colorBy === category)}
                fill={getColorForCategory(category)}
              />
            ))
          )}
          {colorByFeature !== 'none' && <Legend />}
        </ScatterChart>
      </ResponsiveContainer>
    )
  }

  // Loading state
  if (loading || loadingExtra) {
    return (
      <Center h={400}>
        <Stack align="center" gap="md">
          <Loader size="lg" color="grape" type="dots" />
          <Text c="dimmed">Loading data explorer...</Text>
        </Stack>
      </Center>
    )
  }

  return (
    <Stack gap="lg">
      {/* View Mode Tabs */}
      <Tabs value={viewMode} onChange={(v) => { setViewMode(v as ViewMode); logQuery(`Switched to ${v} view`) }}>
        <Tabs.List>
          <Tabs.Tab value="demographics" leftSection={<IconUsers size={16} />}>
            Demographics
          </Tabs.Tab>
          <Tabs.Tab 
            value="nudenet" 
            leftSection={<IconBodyScan size={16} />}
            rightSection={nudenetData ? <Badge size="xs" color="red">{nudenetData.summary.totalDetections}</Badge> : null}
          >
            NudeNet
          </Tabs.Tab>
          <Tabs.Tab 
            value="compare" 
            leftSection={<IconArrowsExchange size={16} />}
            rightSection={explorerData ? <Badge size="xs" color="grape">{explorerData.totalScenes}</Badge> : null}
          >
            X vs Y Compare
          </Tabs.Tab>
        </Tabs.List>

        {/* Demographics View */}
        <Tabs.Panel value="demographics" pt="md">
          {!demographics ? (
            <Paper p="xl" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
              <Stack align="center" gap="md">
                <IconUsers size={48} stroke={1.5} color="#868e96" />
                <Text c="dimmed">Demographics data not available</Text>
              </Stack>
            </Paper>
          ) : (
            <Stack gap="lg">
              {/* Summary Stats */}
              <SimpleGrid cols={{ base: 2, sm: 4 }}>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Group justify="space-between">
                    <Stack gap={2}>
                      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Total Persons</Text>
                      <Text size="xl" fw={700}>{demographics.summary.totalPersonsDetected.toLocaleString()}</Text>
                    </Stack>
                    <ThemeIcon variant="light" color="violet" size="lg" radius="md">
                      <IconUsers size={20} />
                    </ThemeIcon>
                  </Group>
                </Paper>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Group justify="space-between">
                    <Stack gap={2}>
                      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Average Age</Text>
                      <Text size="xl" fw={700}>{demographics.summary.meanAgeOverall?.toFixed(1) || 'N/A'}</Text>
                    </Stack>
                    <ThemeIcon variant="light" color="orange" size="lg" radius="md">
                      <IconBabyCarriage size={20} />
                    </ThemeIcon>
                  </Group>
                </Paper>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Group justify="space-between">
                    <Stack gap={2}>
                      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Scenes Analyzed</Text>
                      <Text size="xl" fw={700}>{demographics.summary.scenesWithPersons} / {demographics.summary.totalScenes}</Text>
                    </Stack>
                    <ThemeIcon variant="light" color="cyan" size="lg" radius="md">
                      <IconUserScan size={20} />
                    </ThemeIcon>
                  </Group>
                </Paper>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Group justify="space-between">
                    <Stack gap={2}>
                      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Gender Ratio</Text>
                      <Text size="xl" fw={700}>
                        {demographics.genderDistribution.Male + demographics.genderDistribution.Female > 0
                          ? `${Math.round((demographics.genderDistribution.Female / (demographics.genderDistribution.Male + demographics.genderDistribution.Female)) * 100)}% F`
                          : 'N/A'}
                      </Text>
                    </Stack>
                    <Group gap={4}>
                      <IconMan size={18} color="#4fc3f7" />
                      <IconWoman size={18} color="#f06292" />
                    </Group>
                  </Group>
                </Paper>
              </SimpleGrid>

              {/* Chart Controls */}
              <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                <Group justify="space-between" mb="md">
                  <Group gap="md">
                    <Select
                      value={demoMetric}
                      onChange={(v) => { setDemoMetric(v || 'ageDistribution'); logQuery(`Demographics metric: ${v}`) }}
                      data={[
                        { value: 'ageDistribution', label: 'Age Categories' },
                        { value: 'genderDistribution', label: 'Gender Distribution' },
                        { value: 'raceDistribution', label: 'Race Distribution' },
                        { value: 'ageHistogram', label: 'Age Histogram' },
                      ]}
                      w={200}
                      styles={{ input: { backgroundColor: '#25262b', borderColor: '#373A40' } }}
                    />
                    <SegmentedControl
                      value={demoChartType}
                      onChange={(v) => setDemoChartType(v)}
                      data={[
                        { value: 'bar', label: <IconChartBar size={16} /> },
                        { value: 'pie', label: <IconChartPie size={16} /> },
                      ]}
                      styles={{ root: { backgroundColor: '#25262b' } }}
                    />
                  </Group>
                  <Badge variant="light" color="violet">{demographicsChartData.length} items</Badge>
                </Group>
                {renderChart(demographicsChartData, demoChartType)}
              </Paper>
            </Stack>
          )}
        </Tabs.Panel>

        {/* NudeNet View */}
        <Tabs.Panel value="nudenet" pt="md">
          {!nudenetData ? (
            <Paper p="xl" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
              <Stack align="center" gap="md">
                <IconBodyScan size={48} stroke={1.5} color="#868e96" />
                <Text c="dimmed">NudeNet data not available</Text>
                <Text c="dimmed" size="sm">Run nudenet_batch_processor.py to generate this data</Text>
              </Stack>
            </Paper>
          ) : (
            <Stack gap="lg">
              {/* Summary Stats */}
              <SimpleGrid cols={{ base: 2, sm: 4 }}>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Group justify="space-between">
                    <Stack gap={2}>
                      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Total Detections</Text>
                      <Text size="xl" fw={700}>{nudenetData.summary.totalDetections.toLocaleString()}</Text>
                    </Stack>
                    <ThemeIcon variant="light" color="red" size="lg" radius="md">
                      <IconBodyScan size={20} />
                    </ThemeIcon>
                  </Group>
                </Paper>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Group justify="space-between">
                    <Stack gap={2}>
                      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Scenes with Content</Text>
                      <Text size="xl" fw={700}>{nudenetData.summary.scenesWithDetections} / {nudenetData.summary.totalScenes}</Text>
                    </Stack>
                    <RingProgress
                      size={50}
                      thickness={5}
                      roundCaps
                      sections={[{ value: (nudenetData.summary.scenesWithDetections / nudenetData.summary.totalScenes) * 100, color: 'red' }]}
                    />
                  </Group>
                </Paper>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Group justify="space-between">
                    <Stack gap={2}>
                      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Scenes with Exposed</Text>
                      <Text size="xl" fw={700}>{nudenetData.summary.scenesWithExposed}</Text>
                    </Stack>
                    <ThemeIcon variant="light" color="orange" size="lg" radius="md">
                      <IconEye size={20} />
                    </ThemeIcon>
                  </Group>
                </Paper>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Group justify="space-between">
                    <Stack gap={2}>
                      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Avg Detection Rate</Text>
                      <Text size="xl" fw={700}>{(nudenetData.summary.avgDetectionRate * 100).toFixed(1)}%</Text>
                    </Stack>
                    <ThemeIcon variant="light" color="teal" size="lg" radius="md">
                      <IconTrendingUp size={20} />
                    </ThemeIcon>
                  </Group>
                </Paper>
              </SimpleGrid>

              {/* Chart Controls */}
              <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                <Group justify="space-between" mb="md">
                  <Group gap="md">
                    <Select
                      value={nudenetMetric}
                      onChange={(v) => { setNudenetMetric(v || 'labelDistribution'); logQuery(`NudeNet metric: ${v}`) }}
                      data={[
                        { value: 'labelDistribution', label: 'Body Part Labels' },
                        { value: 'exposedVsCovered', label: 'Exposed vs Covered' },
                        { value: 'detectionRateHistogram', label: 'Detection Rate Distribution' },
                      ]}
                      w={220}
                      styles={{ input: { backgroundColor: '#25262b', borderColor: '#373A40' } }}
                    />
                    {nudenetMetric === 'labelDistribution' && (
                      <Switch
                        checked={showExposedOnly}
                        onChange={(e) => setShowExposedOnly(e.currentTarget.checked)}
                        label="Exposed only"
                        color="red"
                      />
                    )}
                  </Group>
                  <Group gap="xs">
                    <ColorSwatch color={COLORS.nudenet.exposed} size={16} />
                    <Text size="xs" c="dimmed">Exposed</Text>
                    <ColorSwatch color={COLORS.nudenet.covered} size={16} />
                    <Text size="xs" c="dimmed">Covered</Text>
                  </Group>
                </Group>
                {renderChart(nudenetChartData, nudenetMetric === 'exposedVsCovered' ? 'pie' : 'bar', 400)}
              </Paper>

              {/* Label breakdown */}
              <SimpleGrid cols={{ base: 1, md: 2 }}>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Text size="sm" fw={600} mb="md" c="red.4">Exposed Content Breakdown</Text>
                  <Stack gap="xs">
                    {Object.entries(nudenetData.exposedDistribution)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 8)
                      .map(([label, count]) => {
                        const total = Object.values(nudenetData.exposedDistribution).reduce((a, b) => a + b, 0)
                        const pct = total > 0 ? (count / total) * 100 : 0
                        return (
                          <Group key={label} justify="space-between">
                            <Text size="sm">{NUDENET_LABEL_NAMES[label] || label}</Text>
                            <Group gap="xs">
                              <Text size="sm" fw={600}>{count}</Text>
                              <Text size="xs" c="dimmed">({pct.toFixed(1)}%)</Text>
                            </Group>
                          </Group>
                        )
                      })}
                  </Stack>
                </Paper>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Text size="sm" fw={600} mb="md" c="green.4">Covered Content Breakdown</Text>
                  <Stack gap="xs">
                    {Object.entries(nudenetData.coveredDistribution)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 8)
                      .map(([label, count]) => {
                        const total = Object.values(nudenetData.coveredDistribution).reduce((a, b) => a + b, 0)
                        const pct = total > 0 ? (count / total) * 100 : 0
                        return (
                          <Group key={label} justify="space-between">
                            <Text size="sm">{NUDENET_LABEL_NAMES[label] || label}</Text>
                            <Group gap="xs">
                              <Text size="sm" fw={600}>{count}</Text>
                              <Text size="xs" c="dimmed">({pct.toFixed(1)}%)</Text>
                            </Group>
                          </Group>
                        )
                      })}
                  </Stack>
                </Paper>
              </SimpleGrid>
            </Stack>
          )}
        </Tabs.Panel>

        {/* X vs Y Comparison View */}
        <Tabs.Panel value="compare" pt="md">
          {!explorerData ? (
            <Paper p="xl" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
              <Stack align="center" gap="md">
                <IconArrowsExchange size={48} stroke={1.5} color="#868e96" />
                <Text c="dimmed">Unified data not available</Text>
              </Stack>
            </Paper>
          ) : (
            <Stack gap="lg">
              {/* Feature Selection Controls */}
              <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                <Group justify="space-between" mb="md">
                  <Group gap="xs">
                    <ThemeIcon variant="light" color="grape" size="sm">
                      <IconAdjustments size={14} />
                    </ThemeIcon>
                    <Text size="sm" fw={600}>Feature Comparison</Text>
                  </Group>
                  {correlation !== null && (
                    <Badge 
                      variant="light" 
                      color={Math.abs(correlation) > 0.5 ? 'green' : Math.abs(correlation) > 0.3 ? 'yellow' : 'gray'}
                      size="lg"
                    >
                      r = {correlation.toFixed(3)}
                    </Badge>
                  )}
                </Group>
                
                <SimpleGrid cols={{ base: 1, sm: 2, md: 4 }} mb="md">
                  <Select
                    label="X-Axis Feature"
                    value={xFeature}
                    onChange={(v) => { setXFeature(v || 'meanAge'); logQuery(`X-axis: ${v}`) }}
                    data={featureOptions}
                    searchable
                    styles={{ input: { backgroundColor: '#25262b', borderColor: '#373A40' } }}
                  />
                  <Select
                    label="Y-Axis Feature"
                    value={yFeature}
                    onChange={(v) => { setYFeature(v || 'nudenetDetectionRate'); logQuery(`Y-axis: ${v}`) }}
                    data={featureOptions}
                    searchable
                    styles={{ input: { backgroundColor: '#25262b', borderColor: '#373A40' } }}
                  />
                  <Select
                    label="Color By"
                    value={colorByFeature}
                    onChange={(v) => { setColorByFeature(v || 'none'); logQuery(`Color by: ${v}`) }}
                    data={colorByOptions}
                    styles={{ input: { backgroundColor: '#25262b', borderColor: '#373A40' } }}
                  />
                  <Stack gap="xs" pt={24}>
                    <Switch
                      checked={filterPersonPresent}
                      onChange={(e) => setFilterPersonPresent(e.currentTarget.checked)}
                      label="Only scenes with person"
                      color="grape"
                    />
                  </Stack>
                </SimpleGrid>

                <Divider my="sm" color="dark.5" />
                
                <Group justify="space-between">
                  <Text size="xs" c="dimmed">
                    Showing {comparisonData.length} of {explorerData.totalScenes} scenes
                  </Text>
                  <Button
                    variant="subtle"
                    size="xs"
                    leftSection={<IconRefresh size={14} />}
                    onClick={() => {
                      setXFeature('meanAge')
                      setYFeature('nudenetDetectionRate')
                      setColorByFeature('dominantEmotion')
                      setFilterPersonPresent(false)
                      logQuery('Reset to defaults')
                    }}
                  >
                    Reset
                  </Button>
                </Group>
              </Paper>

              {/* Scatter Plot */}
              <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                <Group justify="space-between" mb="md">
                  <Text size="sm" fw={600}>
                    {featureOptions.find(f => f.value === xFeature)?.label} vs {featureOptions.find(f => f.value === yFeature)?.label}
                  </Text>
                  <Badge variant="light" color="grape">{comparisonData.length} points</Badge>
                </Group>
                {renderComparisonScatter()}
              </Paper>

              {/* Quick Stats */}
              <SimpleGrid cols={{ base: 2, sm: 4 }}>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Text size="xs" c="dimmed" tt="uppercase" fw={600}>X Mean</Text>
                  <Text size="lg" fw={700}>
                    {comparisonData.length > 0 
                      ? (comparisonData.reduce((a, b) => a + b.x, 0) / comparisonData.length).toFixed(2)
                      : 'N/A'}
                  </Text>
                </Paper>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Y Mean</Text>
                  <Text size="lg" fw={700}>
                    {comparisonData.length > 0 
                      ? (comparisonData.reduce((a, b) => a + b.y, 0) / comparisonData.length).toFixed(2)
                      : 'N/A'}
                  </Text>
                </Paper>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Text size="xs" c="dimmed" tt="uppercase" fw={600}>X Range</Text>
                  <Text size="lg" fw={700}>
                    {comparisonData.length > 0 
                      ? `${Math.min(...comparisonData.map(d => d.x)).toFixed(1)} - ${Math.max(...comparisonData.map(d => d.x)).toFixed(1)}`
                      : 'N/A'}
                  </Text>
                </Paper>
                <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
                  <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Y Range</Text>
                  <Text size="lg" fw={700}>
                    {comparisonData.length > 0 
                      ? `${Math.min(...comparisonData.map(d => d.y)).toFixed(1)} - ${Math.max(...comparisonData.map(d => d.y)).toFixed(1)}`
                      : 'N/A'}
                  </Text>
                </Paper>
              </SimpleGrid>
            </Stack>
          )}
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
        <ScrollArea h={80}>
          <Stack gap={2}>
            {queryLog.slice(-10).map((log, i) => (
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
