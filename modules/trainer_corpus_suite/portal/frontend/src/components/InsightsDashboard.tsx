import { useMemo } from 'react'
import {
  Paper,
  SimpleGrid,
  Text,
  Group,
  Stack,
  Box,
  RingProgress,
  Center,
  ThemeIcon,
  rem
} from '@mantine/core'
import {
  IconVideo,
  IconScissors,
  IconUser,
  IconMessage,
  IconMoodSmile,
  IconActivity,
  IconClock,
  IconAlertTriangle
} from '@tabler/icons-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  Legend
} from 'recharts'
import type { PipelineInsights, ClipData } from '../types'

const EMOTION_COLORS: Record<string, string> = {
  happy: '#40c057',
  sad: '#748ffc',
  angry: '#ff6b6b',
  fear: '#f59f00',
  surprise: '#da77f2',
  disgust: '#20c997',
  neutral: '#868e96',
  unknown: '#495057'
}

interface InsightsDashboardProps {
  insights: PipelineInsights
  clips: ClipData[]
}

interface StatCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  color: string
}

function StatCard({ title, value, subtitle, icon, color }: StatCardProps) {
  return (
    <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }} className="stat-card">
      <Group justify="space-between" mb="xs">
        <Text size="xs" c="dimmed" tt="uppercase" fw={600} style={{ letterSpacing: '0.5px' }}>
          {title}
        </Text>
        <ThemeIcon variant="light" color={color} size="md" radius="md">
          {icon}
        </ThemeIcon>
      </Group>
      <Text size="xl" fw={700} style={{ fontSize: rem(28) }}>
        {value}
      </Text>
      {subtitle && (
        <Text size="xs" c="dimmed" mt={4}>
          {subtitle}
        </Text>
      )}
    </Paper>
  )
}

export function InsightsDashboard({ insights, clips }: InsightsDashboardProps) {
  // Prepare emotion bar chart data
  const emotionBarData = useMemo(() => {
    return Object.entries(insights.emotionDistribution)
      .map(([emotion, count]) => ({
        name: emotion,
        count,
        fill: EMOTION_COLORS[emotion] || EMOTION_COLORS.unknown
      }))
      .sort((a, b) => b.count - a.count)
  }, [insights.emotionDistribution])

  // Prepare pie chart data
  const personPieData = useMemo(() => [
    { name: 'With Person', value: insights.clipsWithPerson, color: '#40c057' },
    { name: 'Without Person', value: insights.totalClips - insights.clipsWithPerson, color: '#495057' }
  ], [insights])

  // Prepare valence/arousal scatter data
  const valenceArousalData = useMemo(() => {
    return clips
      .filter(c => c.personPresent && c.meanValence !== undefined)
      .map(c => ({
        valence: c.meanValence,
        arousal: c.meanArousal,
        emotion: c.dominantEmotion,
        name: c.sceneName
      }))
  }, [clips])

  // Caption success rate
  const captionSuccessRate = useMemo(() => {
    if (insights.totalClips === 0) return 0
    return (insights.clipsWithCaption / insights.totalClips) * 100
  }, [insights])

  return (
    <Stack gap="lg">
      {/* Key Metrics */}
      <SimpleGrid cols={{ base: 2, sm: 4 }}>
        <StatCard
          title="Total Videos"
          value={insights.totalVideos}
          icon={<IconVideo size={18} />}
          color="cerulean"
        />
        <StatCard
          title="Total Clips"
          value={insights.totalClips}
          subtitle={`${insights.clipsWithPerson} with person detected`}
          icon={<IconScissors size={18} />}
          color="grape"
        />
        <StatCard
          title="Captions Generated"
          value={insights.clipsWithCaption}
          subtitle={`${insights.clipsFailed} failed`}
          icon={<IconMessage size={18} />}
          color="teal"
        />
        <StatCard
          title="Processing Time"
          value={`${(insights.processingTime / 60).toFixed(1)}m`}
          subtitle={new Date(insights.processedAt).toLocaleDateString()}
          icon={<IconClock size={18} />}
          color="orange"
        />
      </SimpleGrid>

      {/* Second Row: More metrics */}
      <SimpleGrid cols={{ base: 2, sm: 4 }}>
        <StatCard
          title="Avg Valence"
          value={insights.averageValence.toFixed(2)}
          subtitle={insights.averageValence > 0 ? 'Positive bias' : 'Negative bias'}
          icon={<IconMoodSmile size={18} />}
          color={insights.averageValence > 0 ? 'green' : 'red'}
        />
        <StatCard
          title="Avg Arousal"
          value={insights.averageArousal.toFixed(2)}
          subtitle={insights.averageArousal > 0.5 ? 'High energy' : 'Low energy'}
          icon={<IconActivity size={18} />}
          color="yellow"
        />
        <StatCard
          title="Avg Coverage"
          value={`${(insights.averageDetectionCoverage * 100).toFixed(1)}%`}
          subtitle="Person detection coverage"
          icon={<IconUser size={18} />}
          color="indigo"
        />
        <StatCard
          title="Caption Rate"
          value={`${captionSuccessRate.toFixed(1)}%`}
          subtitle={insights.clipsFailed > 0 ? `${insights.clipsFailed} errors` : 'All successful'}
          icon={insights.clipsFailed > 0 ? <IconAlertTriangle size={18} /> : <IconMessage size={18} />}
          color={captionSuccessRate > 80 ? 'green' : captionSuccessRate > 50 ? 'yellow' : 'red'}
        />
      </SimpleGrid>

      {/* Charts Row */}
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        {/* Emotion Distribution Bar Chart */}
        <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
          <Text size="sm" fw={600} mb="md">Emotion Distribution</Text>
          <Box h={280}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={emotionBarData} layout="vertical" margin={{ left: 0, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2c2e33" />
                <XAxis type="number" stroke="#868e96" fontSize={12} />
                <YAxis
                  type="category"
                  dataKey="name"
                  stroke="#868e96"
                  fontSize={12}
                  width={70}
                  tickFormatter={(v) => v.charAt(0).toUpperCase() + v.slice(1)}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1b1e',
                    border: '1px solid #2c2e33',
                    borderRadius: 8,
                    color: '#fff'
                  }}
                />
                <Bar dataKey="count" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Paper>

        {/* Person Detection Pie */}
        <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
          <Text size="sm" fw={600} mb="md">Person Detection</Text>
          <Box h={280}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={personPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  labelLine={{ stroke: '#868e96' }}
                >
                  {personPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1b1e',
                    border: '1px solid #2c2e33',
                    borderRadius: 8,
                    color: '#fff'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        </Paper>
      </SimpleGrid>

      {/* Valence/Arousal Scatter Plot */}
      <Paper p="md" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
        <Text size="sm" fw={600} mb="md">Valence vs Arousal Distribution</Text>
        <Box h={350}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2c2e33" />
              <XAxis
                type="number"
                dataKey="valence"
                name="Valence"
                stroke="#868e96"
                fontSize={12}
                domain={[-1, 1]}
                label={{ value: 'Valence (Negative ← → Positive)', position: 'bottom', fill: '#868e96', fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="arousal"
                name="Arousal"
                stroke="#868e96"
                fontSize={12}
                domain={[0, 1]}
                label={{ value: 'Arousal', angle: -90, position: 'insideLeft', fill: '#868e96', fontSize: 11 }}
              />
              <ZAxis range={[50, 200]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1b1e',
                  border: '1px solid #2c2e33',
                  borderRadius: 8,
                  color: '#fff'
                }}
                formatter={(value: number) => value.toFixed(3)}
              />
              <Legend />
              {Object.keys(EMOTION_COLORS).filter(e => e !== 'unknown').map(emotion => (
                <Scatter
                  key={emotion}
                  name={emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                  data={valenceArousalData.filter(d => d.emotion === emotion)}
                  fill={EMOTION_COLORS[emotion]}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </Box>
      </Paper>

      {/* Summary Ring Progress Cards */}
      <SimpleGrid cols={{ base: 1, sm: 3 }}>
        <Paper p="lg" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
          <Group justify="space-between" align="flex-start">
            <Stack gap={4}>
              <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Person Detection Rate</Text>
              <Text size="xl" fw={700}>
                {((insights.clipsWithPerson / insights.totalClips) * 100).toFixed(1)}%
              </Text>
              <Text size="xs" c="dimmed">
                {insights.clipsWithPerson} / {insights.totalClips} clips
              </Text>
            </Stack>
            <RingProgress
              size={80}
              thickness={8}
              roundCaps
              sections={[
                { value: (insights.clipsWithPerson / insights.totalClips) * 100, color: 'green' }
              ]}
            />
          </Group>
        </Paper>

        <Paper p="lg" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
          <Group justify="space-between" align="flex-start">
            <Stack gap={4}>
              <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Caption Success</Text>
              <Text size="xl" fw={700}>
                {captionSuccessRate.toFixed(1)}%
              </Text>
              <Text size="xs" c="dimmed">
                {insights.clipsWithCaption} / {insights.totalClips} clips
              </Text>
            </Stack>
            <RingProgress
              size={80}
              thickness={8}
              roundCaps
              sections={[
                { value: captionSuccessRate, color: captionSuccessRate > 80 ? 'teal' : 'yellow' }
              ]}
            />
          </Group>
        </Paper>

        <Paper p="lg" radius="md" bg="dark.8" withBorder style={{ borderColor: '#2c2e33' }}>
          <Group justify="space-between" align="flex-start">
            <Stack gap={4}>
              <Text size="xs" c="dimmed" tt="uppercase" fw={600}>Detection Coverage</Text>
              <Text size="xl" fw={700}>
                {(insights.averageDetectionCoverage * 100).toFixed(1)}%
              </Text>
              <Text size="xs" c="dimmed">
                Average across all clips
              </Text>
            </Stack>
            <RingProgress
              size={80}
              thickness={8}
              roundCaps
              sections={[
                { value: insights.averageDetectionCoverage * 100, color: 'cerulean' }
              ]}
            />
          </Group>
        </Paper>
      </SimpleGrid>
    </Stack>
  )
}
